# ==========================================================================
# Master Thesis - Deterministic Planner Renderer
#   - André Filipe Gomes Silvestre, 20240502
#
#   Renders structured PlannerAgent output into LISBOA Markdown using a stable
#   visual contract. This module owns headings, labels, indentation, section
#   order, source footers, and cleanup so the planner LLM does not need to emit
#   final Markdown directly.
# ==========================================================================

import re
from datetime import datetime, timedelta
from typing import Dict, Iterable, List

from agent.planning.models import PlanDraft, SourceRef


_BLOCK_KIND_EMOJI = {
    "activity": "📍",
    "place": "📍",
    "museum": "🏛️",
    "culture": "🏛️",
    "event": "🎭",
    "food": "🍽️",
    "coffee": "☕",
    "pastry": "🥐",
    "transport": "🚇",
    "walk": "🚶",
    "service": "🏛️",
}


def render_plan_markdown(draft: PlanDraft, sources: Dict[str, SourceRef], language: str = "en") -> str:
    """Render a plan draft using the LISBOA visual response contract.

    Args:
        draft: Structured plan draft produced by PlannerAgent.
        sources: Public source references available for source-footers.
        language: Detected or requested response language.

    Returns:
        Markdown answer ready for Streamlit rendering.
    """
    is_pt = (language or "en").lower().startswith("pt")
    title = _clean_inline(draft.title) or ("Plano de Lisboa" if is_pt else "Lisbon Plan")
    direct = _clean_inline(draft.direct_answer) or (
        "Plano limitado aos dados confirmados disponíveis." if is_pt else "Plan limited to the confirmed data available."
    )

    lines: List[str] = [f"### 📅 **{title}**", "", f"✅ **{'Resposta direta' if is_pt else 'Direct answer'}:** {direct}"]

    constraint_items = [
        item for item in _clean_list(draft.constraints_used, max_items=3)
        if not _is_generic_planner_note(item)
    ]
    if constraint_items:
        lines.extend(["", "", f"💡 **{'Dicas' if is_pt else 'Tips'}:**"])
        for item in constraint_items:
            lines.append(f"    - {item}")

    visible_blocks = draft.blocks[:8]
    if visible_blocks:
        lines.extend(["", "---", "", f"### 📍 **{'Roteiro sugerido' if is_pt else 'Suggested route'}**"])
        for index, block in enumerate(visible_blocks, start=1):
            block_title = _clean_inline(block.title) or (f"Bloco {index}" if is_pt else f"Block {index}")
            block_title = re.sub(r"^(?:Block|Bloco)\s*\d+\s*[·:-]\s*", "", block_title, flags=re.IGNORECASE).strip()
            lines.extend(["", f"**🏷️ {block_title}**"])
            if block.purpose:
                lines.append(f"    - 📝 {_clean_inline(block.purpose)}")
            for detail in _clean_list(block.details, max_items=9):
                lines.append(_format_detail_bullet(detail, is_pt))
            for movement in _clean_list(block.movement, max_items=3):
                lines.append(_format_movement_bullet(movement, is_pt=is_pt, indent="    "))
            for weather in _clean_list(block.weather, max_items=2):
                lines.append(f"    - ☔ {weather}")
            for limitation in _clean_list(block.limitations, max_items=2):
                lines.append(f"    - ⚠️ {limitation}")

    block_titles = [_clean_inline(block.title) for block in draft.blocks if block.title]
    movement_items = _filter_movement_items(
        _clean_list(draft.movement_logic, max_items=6),
        block_titles=block_titles,
    )
    movement_items = _ensure_adjacent_short_walks(movement_items, visible_blocks, is_pt=is_pt)
    weather_items = [
        item for item in _clean_list(draft.weather_strategy, max_items=6)
        if not _is_placeholder_weather_item(item)
    ]
    limitation_items = _clean_list(draft.limitations, max_items=6) or [
        "horários, bilhetes, preços, reservas e disponibilidade em tempo real só estão confirmados quando indicados acima"
        if is_pt
        else "opening hours, tickets, prices, bookings, and live availability are confirmed only where stated above"
    ]

    _append_movement_section(lines, movement_items, is_pt=is_pt)
    _append_section(lines, "☔", "Adaptação ao tempo" if is_pt else "Weather adaptation", weather_items)
    _append_section(
        lines,
        "💡",
        "Dicas" if is_pt else "Tips",
        [item for item in draft.tips if not _is_generic_planner_note(item)],
    )
    _append_section(lines, "⚠️", "Notas finais" if is_pt else "Final notes", limitation_items)

    footer = _source_footer(draft, sources, is_pt, rendered_body="\n".join(lines))
    if footer:
        lines.extend(["", footer])
    return _clean_markdown("\n".join(lines))


def _append_section(lines: List[str], emoji: str, title: str, items: Iterable[str]) -> None:
    """Append a labeled Markdown section when it has valid items.

    Args:
        lines: Mutable Markdown line buffer.
        emoji: Section emoji used by the response contract.
        title: Section heading text.
        items: Candidate bullet items.
    """
    cleaned = _clean_list(list(items), max_items=6)
    if not cleaned:
        return
    lines.extend(["", "---", "", f"### {emoji} **{title}**"])
    for item in cleaned:
        lines.append(f"- {item}")


def _append_movement_section(lines: List[str], items: Iterable[str], *, is_pt: bool) -> None:
    """Append movement guidance with the same visual field style as route tools."""
    cleaned = _clean_list(list(items), max_items=6)
    if not cleaned:
        return
    emoji = _movement_section_emoji(cleaned)
    lines.extend(["", "---", "", f"### {emoji} **{'Como te deslocas' if is_pt else 'How to move'}**"])
    for item in cleaned:
        lines.append(_format_movement_bullet(item, is_pt=is_pt, indent=""))


def _movement_section_emoji(items: Iterable[str]) -> str:
    """Choose a movement-section icon that matches the visible guidance."""
    normalized = _normalize_for_match(" ".join(str(item or "") for item in items))
    transport_signal = re.search(
        r"\b(?:metro|carris|autocarro|bus|tram|eletrico|elétrico|train|comboio|cp|linha|line)\b",
        normalized,
    )
    walk_signal = re.search(r"\b(?:walk|walking|caminh\w*|andar|pe|pé)\b", normalized)
    return "🚶" if walk_signal and not transport_signal else "🚇"


def _format_movement_bullet(item: str, *, is_pt: bool, indent: str = "    ") -> str:
    """Render one movement item as an icon-labelled nested bullet."""
    text = _clean_inline(item)
    text_for_match = re.sub(
        r"^[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s+",
        "",
        text,
    ).strip()
    bold_match = re.match(
        r"^\*\*(?P<label>[A-Za-zÀ-ÿ0-9 /'-]{2,70})\s*:\*\*\s*(?P<value>.+)$",
        text_for_match,
    )
    route_bold_match = re.match(
        r"^\*\*(?P<route>[^*\n]{2,160}(?:→|->)[^*\n]{2,160})\s*:\*\*\s*(?P<value>.+)$",
        text_for_match,
    )
    if route_bold_match:
        route = route_bold_match.group("route").strip()
        value = route_bold_match.group("value").strip()
        icon = "⚠️" if re.match(r"^⚠️", text) or re.search(r"\b(?:not confirmed|não ficou confirmada|nao ficou confirmada)\b", text, re.IGNORECASE) else _movement_icon_for_text(text)
        return f"{indent}- {icon} **{route}:** {value}"
    match = bold_match or re.match(r"^(?P<label>[A-Za-zÀ-ÿ0-9 /'-]{2,70})\s*:\s*(?P<value>.+)$", text_for_match)
    if not match:
        bold_heading = re.match(r"^\*\*(?P<value>[^*\n]{2,120})\*\*$", text_for_match)
        heading_text = bold_heading.group("value").strip() if bold_heading else text_for_match
        move_match = re.match(r"^(?P<action>Board at|Transfer at|Exit at|Continue on|Start at|Embarque em|Transferência em|Transferencia em|Saia em|Sair em|Continuar em|Começar em|Comecar em)\s+(?P<place>.+)$", heading_text, flags=re.IGNORECASE)
        if move_match:
            action = move_match.group("action").strip()
            place = move_match.group("place").strip()
            normalized_action = _normalize_for_match(action)
            action_map = {
                "board at": ("📍", "Embarque" if is_pt else "Board"),
                "start at": ("📍", "Início" if is_pt else "Start"),
                "transfer at": ("🔁", "Transbordo" if is_pt else "Transfer"),
                "continue on": ("🚇", "Continuação" if is_pt else "Continue"),
                "exit at": ("📍", "Saída" if is_pt else "Exit"),
                "embarque em": ("📍", "Embarque" if is_pt else "Board"),
                "transferencia em": ("🔁", "Transbordo" if is_pt else "Transfer"),
                "saia em": ("📍", "Saída" if is_pt else "Exit"),
                "sair em": ("📍", "Saída" if is_pt else "Exit"),
                "continuar em": ("🚇", "Continuação" if is_pt else "Continue"),
                "comecar em": ("📍", "Início" if is_pt else "Start"),
            }
            icon, label = action_map.get(normalized_action, ("🚇", action))
            return f"{indent}- {icon} **{label}:** {place}"
        route_match = re.match(
            r"^(?P<route>[^:\n]{2,140}(?:→|->)[^:\n]{2,140})\s*:\s*(?P<value>.+)$",
            heading_text,
        )
        if route_match:
            icon = _movement_icon_for_text(text)
            route = route_match.group("route").strip()
            value = route_match.group("value").strip()
            return f"{indent}- {icon} **{route}:** {value}"
        if re.match(r"^[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s+", text):
            return f"{indent}- {text}"
        icon = _movement_icon_for_text(text)
        return f"{indent}- {icon} {text}"

    label = match.group("label").strip()
    value = match.group("value").strip(" *")
    normalized = _normalize_for_match(label)
    label_map = {
        "best transport": ("🚇", "Melhor transporte" if is_pt else "Best transport"),
        "best realistic option": ("🚇", "Melhor opção realista" if is_pt else "Best realistic option"),
        "best supported route": ("🚇", "Melhor percurso confirmado" if is_pt else "Best supported route"),
        "best supported option": ("🚇", "Melhor opção confirmada" if is_pt else "Best supported option"),
        "estimated total time": ("⏱️", "Tempo total estimado" if is_pt else "Estimated total time"),
        "estimated travel time": ("⏱️", "Tempo de viagem estimado" if is_pt else "Estimated travel time"),
        "estimated total travel time": ("⏱️", "Tempo total de viagem estimado" if is_pt else "Estimated total travel time"),
        "estimated metro time": ("⏱️", "Tempo de metro estimado" if is_pt else "Estimated metro time"),
        "route": ("🗺️", "Percurso" if is_pt else "Route"),
        "walk": ("🚶", "Caminhada" if is_pt else "Walk"),
        "transfer": ("🔁", "Transbordo" if is_pt else "Transfer"),
        "cultural stop": ("🏛️", "Paragem cultural" if is_pt else "Cultural stop"),
    }
    if normalized.startswith("next metro"):
        icon, display_label = "🚇", label
    else:
        icon, display_label = label_map.get(normalized, (_movement_icon_for_text(text), label))
    return f"{indent}- {icon} **{display_label}:** {value}"


def _ensure_adjacent_short_walks(items: List[str], blocks: List[object], *, is_pt: bool) -> List[str]:
    """Add missing short walking legs between adjacent same-zone plan blocks."""
    output = list(items or [])
    existing_norm = "\n".join(_normalize_for_match(item) for item in output)
    additions: List[str] = []
    for previous, current in zip(blocks, blocks[1:]):
        previous_name = _block_display_name(str(getattr(previous, "title", "") or ""))
        current_name = _block_display_name(str(getattr(current, "title", "") or ""))
        if not previous_name or not current_name:
            continue
        if _normalize_for_match(previous_name) in existing_norm and _normalize_for_match(current_name) in existing_norm:
            continue
        zone = _shared_walkable_zone(previous, current)
        if not zone:
            continue
        current_time = _block_time(str(getattr(current, "title", "") or ""))
        if is_pt:
            if current_time and re.search(r"(?:almo|lunch)", _normalize_for_match(str(getattr(current, "title", "") or ""))):
                depart_window = _departure_window_for_time(current_time)
                departure_text = depart_window or "15-20 min antes"
                suffix = f"; sai cerca de {departure_text} para chegares ao almoço das {current_time}"
            else:
                suffix = "; mantém esta ligação a pé se o tempo permitir"
            additions.append(f"🚶 {previous_name} → {current_name}: caminhada curta na zona {zone}{suffix}.")
        else:
            if current_time and re.search(r"(?:almo|lunch)", _normalize_for_match(str(getattr(current, "title", "") or ""))):
                depart_window = _departure_window_for_time(current_time)
                departure_text = depart_window or "15-20 min before"
                suffix = f"; leave around {departure_text} to arrive for the {current_time} lunch"
            else:
                suffix = "; keep this as a walking leg if conditions allow"
            additions.append(f"🚶 {previous_name} → {current_name}: short walk in the {zone} area{suffix}.")
    return additions + output


def _departure_window_for_time(time_value: str, *, early_minutes: int = 20, late_minutes: int = 15) -> str:
    """Return a short departure window before an HH:MM time string."""
    try:
        target = datetime.strptime(str(time_value).strip(), "%H:%M")
    except (TypeError, ValueError):
        return ""
    early = target - timedelta(minutes=early_minutes)
    late = target - timedelta(minutes=late_minutes)
    return f"{early.strftime('%H:%M')}-{late.strftime('%H:%M')}"


def _block_display_name(title: str) -> str:
    """Extract the venue name from a rendered/planner block title."""
    cleaned = _clean_inline(title)
    cleaned = re.sub(r"^\d{1,2}:\d{2}\s*[·:-]\s*", "", cleaned).strip()
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[1].strip()
    return cleaned.strip(" .·-")


def _block_time(title: str) -> str:
    """Extract HH:MM from a block title when present."""
    match = re.search(r"\b(\d{1,2}:\d{2})\b", title or "")
    return match.group(1) if match else ""


def _block_zone_text(block: object) -> str:
    """Collect title/detail text for same-zone walking heuristics."""
    parts = [str(getattr(block, "title", "") or "")]
    parts.extend(str(item or "") for item in getattr(block, "details", []) or [])
    return _normalize_for_match(" ".join(parts))


def _shared_walkable_zone(previous: object, current: object) -> str:
    """Return a compact zone label when two adjacent blocks are safely walkable."""
    previous_text = _block_zone_text(previous)
    current_text = _block_zone_text(current)
    central_markers = (
        "carmo", "baixa", "chiado", "correeiros", "douradores", "praca do comercio",
        "praça do comercio", "rua augusta", "largo da se", "largo da sé", "se de lisboa",
    )
    belem_markers = (
        "belem", "belém", "brasilia", "brasília", "jeronimos", "jerónimos",
        "torre de belem", "padrao dos descobrimentos", "padrão dos descobrimentos",
    )
    previous_central = any(marker in previous_text for marker in central_markers)
    current_central = any(marker in current_text for marker in central_markers)
    previous_belem = any(marker in previous_text for marker in belem_markers)
    current_belem = any(marker in current_text for marker in belem_markers)
    if previous_central and current_central and not (previous_belem or current_belem):
        return "Baixa/Chiado"
    if previous_belem and current_belem and not (previous_central or current_central):
        return "Belém"
    return ""


def _movement_icon_for_text(text: str) -> str:
    """Choose a compact icon for a movement item."""
    normalized = _normalize_for_match(text)
    if re.search(r"^(?:walking plan|plano a pe|plano a pé)\b", normalized):
        return "🚶"
    if re.search(r"\b(?:transport|transporte)\b", normalized):
        return "🚇"
    if re.search(r"\b(?:walk|walking|caminh\w*|andar)\b", normalized):
        return "🚶"
    if re.search(r"\b(?:bus|carris|autocarro)\b", normalized):
        return "🚌"
    if re.search(r"\b(?:tram|electrico|eletrico|elétrico)\b", normalized):
        return "🚋"
    if re.search(r"\b(?:train|comboio|cp)\b", normalized):
        return "🚆"
    if re.search(r"\b(?:time|tempo|min|minute|minuto)\b", normalized):
        return "⏱️"
    if re.search(r"\b(?:route|percurso|rota)\b", normalized):
        return "🗺️"
    if re.search(r"\b(?:culture|cultural|museum|museu|gallery|galeria)\b", normalized):
        return "🏛️"
    return "🚇"


def _filter_movement_items(items: Iterable[str], *, block_titles: Iterable[str]) -> List[str]:
    """Keep planner movement sections focused on actual mobility evidence.

    Args:
        items: Candidate movement strings from a structured plan draft.
        block_titles: Grounded route-block titles already rendered as stops.

    Returns:
        Movement items after removing duplicate headings and non-transport
        cultural suggestions that belong in the route blocks instead.
    """
    output: List[str] = []
    block_markers = {
        _normalize_for_match(title)
        for title in block_titles
        if _normalize_for_match(title)
    }
    mobility_re = re.compile(
        r"\b(?:metro|bus|tram|train|walk|walking|transfer|board|exit|route|line|"
        r"carris|cp|station|ligacao|liga[cç][aã]o|desloca[cç][aã]o|percurso|trajeto|"
        r"autocarro|el[eé]trico|comboio|caminh\w*|andar|apanh\w*|sair|mudar|linha|esta[cç][aã]o|paragem)\b",
        re.IGNORECASE,
    )
    cultural_re = re.compile(
        r"\b(?:cultural stop|paragem cultural|museum|museu|gallery|galeria|monument|monumento|"
        r"attraction|atra[cç][aã]o|practical cultural|paragem pr[aá]tica)\b",
        re.IGNORECASE,
    )

    for item in items:
        cleaned = _clean_inline(item)
        normalized = _normalize_for_match(cleaned)
        if not normalized:
            continue
        if re.fullmatch(
            r"(?:[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s*)?\*\*[^*\n]{2,80}(?:→|->)[^*\n]{2,80}\*\*",
            cleaned,
        ):
            continue
        if normalized in {"how to move", "como te deslocas", "movement logic", "logica de movimento"}:
            continue
        if re.fullmatch(r"(?:rota|route|percurso)\s+de\s+um\s+local\s+para\s+outro", normalized):
            continue
        if re.match(r"^(?:estado atual|current status)\b", normalized) and re.search(r"\b(?:metro|carris|cp)\b", normalized):
            continue
        if normalized in {"metro", "metro de lisboa"}:
            continue
        if normalized in {"carris", "carris urbana", "carris urban", "carris metropolitana", "cp"}:
            continue
        if re.fullmatch(
            r"(?:metro(?: de lisboa)?|transportes?|mobility|mobilidade)\s*:?\s*"
            r"(?:ok|normal|circulacao normal|circulação normal|sem alertas|no alerts)",
            normalized,
        ):
            continue
        if re.fullmatch(
            r"(?:circulacao normal|circulação normal|normal service)\s+(?:em|on)\s+(?:todas|all)\s+(?:as\s+)?(?:linhas|lines)",
            normalized,
        ):
            continue
        if re.search(
            r"\b(?:confirma|confirm|check)\b.*\b(?:melhor ligacao|melhor ligação|best connection|operator|operador)\b",
            normalized,
        ):
            continue
        if re.search(
            r"\b(?:diz-me|indica|send|tell me|provide)\b.*\b(?:origem|origin)\b.*\b(?:destino|destination)\b",
            normalized,
        ):
            continue
        if re.fullmatch(
            r"(?:onde comecas|onde começas|where you start|where you are starting|ponto de partida|origem|"
            r"para onde queres ir|para onde quer ir|where you want to go|destination|destino)",
            normalized,
        ):
            continue
        if re.search(
            r"\b(?:para eu\s+)?(?:optimizar|otimizar|optimize)\b.*\b(?:roteiro|plano|itinerary|plan)\b",
            normalized,
        ):
            continue
        if re.fullmatch(
            r"(?:amarela|azul|verde|vermelha|yellow|blue|green|red)\s*:?\s*(?:ok|normal service|circulacao normal|circulação normal)",
            normalized,
        ):
            continue
        if normalized.startswith("estado geral") and re.search(r"\b(?:ok|normal|circulacao|circulação)\b", normalized):
            continue
        if block_markers and normalized in block_markers:
            continue
        if normalized.startswith(("cultural stop", "paragem cultural")):
            continue
        if cultural_re.search(cleaned) and not mobility_re.search(cleaned):
            continue
        if cleaned not in output:
            output.append(cleaned)
    return output


def _source_footer(
    draft: PlanDraft,
    sources: Dict[str, SourceRef],
    is_pt: bool,
    rendered_body: str = "",
) -> str:
    """Build a source footer from source identifiers materially used in a plan.

    Args:
        draft: Structured plan whose source identifiers are inspected.
        sources: Public source references keyed by source identifier.
        is_pt: Whether the rendered answer is in Portuguese.
        rendered_body: Markdown body before the footer, used to drop sources
            that were available to the planner but not materially used.

    Returns:
        Markdown source footer, or an empty string when no source is available.
    """
    used_ids: List[str] = []
    used_ids.extend(draft.source_ids)
    for block in draft.blocks:
        used_ids.extend(block.source_ids)
    if not used_ids:
        used_ids = list(sources.keys())
    deduped = [source_id for source_id in dict.fromkeys(used_ids) if source_id in sources]
    deduped = [
        source_id for source_id in deduped
        if _source_is_materially_used(source_id, rendered_body)
    ]
    if not deduped:
        return ""
    links = []
    for source_id in deduped[:5]:
        source = sources[source_id]
        label = source.label_pt if is_pt else source.label_en
        url = source.url
        if is_pt and source_id == "visitlisboa_places":
            url = "https://www.visitlisboa.com/pt-pt/locais"
        elif is_pt and source_id == "visitlisboa_events":
            url = "https://www.visitlisboa.com/pt-pt/eventos"
        links.append(f"[*{label}*]({url})")
    timestamp = datetime.now().strftime("%H:%M")
    return f"📌 **{'Fonte' if is_pt else 'Source'}:** {' | '.join(links)} | **{'Atualizado' if is_pt else 'Updated'}:** {timestamp}"


def _rewrite_misrouted_detail(text: str) -> str:
    """Rewrite planner detail strings whose label disagrees with the value's leading emoji.

    The planner LLM occasionally fills the ``description`` field with rating, feature, hours,
    price, contact, or website content that already carries its own field emoji and label
    (for example ``"Description: ⭐ Avaliação: TripAdvisor 3.5/5"``). Render those bullets
    under the correct label instead of collapsing them all into ``📝 Descrição``.
    """
    stripped = text.strip()
    if not stripped:
        return text
    label_match = re.match(r"^(?P<label>[A-Za-zÀ-ÿ ]{2,24})\s*:\s*(?P<value>.+)$", stripped)
    if not label_match:
        return text
    raw_label = label_match.group("label").strip().lower()
    value = label_match.group("value").strip()
    description_labels = {"description", "descrição", "descricao"}
    if raw_label not in description_labels:
        return text
    nested = re.match(
        r"^(?P<emoji>[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF])\s*\*?\*?(?P<inner_label>[A-Za-zÀ-ÿ ]{2,24})\*?\*?\s*:\s*(?P<inner_value>.+)$",
        value,
    )
    if not nested:
        return text
    inner_label = nested.group("inner_label").strip()
    inner_value = nested.group("inner_value").strip(" *")
    return f"{inner_label}: {inner_value}"


def _format_detail_bullet(detail: str, is_pt: bool) -> str:
    """Render a planner detail using the same field emojis as worker cards.

    Args:
        detail: Detail string from the structured planner JSON.
        is_pt: Whether the final answer is Portuguese.

    Returns:
        Indented Markdown bullet with a semantic emoji and localized label.
    """
    text = _clean_inline(detail)
    text = _rewrite_misrouted_detail(text)
    match = re.match(r"^\s*(?P<label>[A-Za-zÀ-ÿ ]{2,24})\s*:\s*(?P<value>.+)$", text)
    if not match:
        return f"    - 📝 {text}"

    raw_label = match.group("label").strip().lower()
    value = match.group("value").strip()
    label_map = {
        "description": ("📝", "Descrição" if is_pt else "Description"),
        "descrição": ("📝", "Descrição" if is_pt else "Description"),
        "descricao": ("📝", "Descrição" if is_pt else "Description"),
        "address": ("📍", "Morada" if is_pt else "Address"),
        "morada": ("📍", "Morada" if is_pt else "Address"),
        "location": ("📍", "Local" if is_pt else "Location"),
        "local": ("📍", "Local" if is_pt else "Location"),
        "venue": ("📍", "Local" if is_pt else "Venue"),
        "when": ("🕒", "Quando" if is_pt else "When"),
        "quando": ("🕒", "Quando" if is_pt else "When"),
        "duration": ("⏱️", "Duração" if is_pt else "Duration"),
        "duração": ("⏱️", "Duração" if is_pt else "Duration"),
        "duracao": ("⏱️", "Duração" if is_pt else "Duration"),
        "hours": ("🕒", "Horário" if is_pt else "Hours"),
        "horário": ("🕒", "Horário" if is_pt else "Hours"),
        "horario": ("🕒", "Horário" if is_pt else "Hours"),
        "price": ("💶", "Preço" if is_pt else "Price"),
        "preço": ("💶", "Preço" if is_pt else "Price"),
        "preco": ("💶", "Preço" if is_pt else "Price"),
        "website": ("🌐", "Website"),
        "tickets": ("🎟️", "Bilhetes" if is_pt else "Tickets"),
        "bilhetes": ("🎟️", "Bilhetes" if is_pt else "Tickets"),
        "category": ("🏷️", "Categoria" if is_pt else "Category"),
        "categoria": ("🏷️", "Categoria" if is_pt else "Category"),
        "phone": ("📞", "Telefone" if is_pt else "Phone"),
        "telefone": ("📞", "Telefone" if is_pt else "Phone"),
        "email": ("✉️", "Email"),
        "e-mail": ("✉️", "Email"),
        "rating": ("⭐", "Avaliação" if is_pt else "Rating"),
        "avaliação": ("⭐", "Avaliação" if is_pt else "Rating"),
        "avaliacao": ("⭐", "Avaliação" if is_pt else "Rating"),
        "suggested time": ("⏱️", "Tempo sugerido" if is_pt else "Suggested time"),
        "tempo sugerido": ("⏱️", "Tempo sugerido" if is_pt else "Suggested time"),
        "more details": ("🔗", "Mais detalhes" if is_pt else "More details"),
        "mais detalhes": ("🔗", "Mais detalhes" if is_pt else "More details"),
        "features": ("✨", "Características" if is_pt else "Features"),
        "características": ("✨", "Características" if is_pt else "Features"),
        "caracteristicas": ("✨", "Características" if is_pt else "Features"),
    }
    emoji, label = label_map.get(raw_label, ("📝", match.group("label").strip()))
    if raw_label == "website":
        value = _normalize_website_link_label(value, is_pt)
    return f"    - {emoji} **{label}:** {value}"


def _normalize_website_link_label(value: str, is_pt: bool) -> str:
    """Correct mismatched markdown labels for official and VisitLisboa links."""
    match = re.match(r"^\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)]+)\)$", value.strip(), re.IGNORECASE)
    if not match:
        return value
    label = match.group("label").strip()
    url = match.group("url").strip()
    if "visitlisboa.com" in url.lower() and label.lower() in {"visitlisboa", "details", "more details", "mais detalhes"}:
        corrected = "VisitLisboa"
    elif label.lower() == "visitlisboa":
        corrected = "Website oficial" if is_pt else "Official website"
    else:
        corrected = label
    return f"[{corrected}]({url})"


def _is_placeholder_weather_item(item: str) -> bool:
    """Return whether a weather item only says weather evidence is absent."""
    normalized = re.sub(r"\s+", " ", (item or "").strip().lower())
    if not normalized:
        return True
    return any(
        marker in normalized
        for marker in (
            "weather was not confirmed",
            "weather was not provided",
            "not confirmed",
            "please verify the current forecast",
            "verify the current forecast",
            "no weather warnings confirmed",
            "no weather-specific",
            "no additional weather",
            "forecast not confirmed",
            "sem adaptação meteorológica",
            "meteorologia não",
            "tempo não confirmado",
            "tempo nao confirmado",
            "sem meteorologia confirmada",
            "previsão não confirmada",
            "previsao nao confirmada",
        )
    )


def _is_generic_planner_note(item: str) -> bool:
    """Return whether a planner note is too generic to help the user."""
    normalized = re.sub(r"\s+", " ", (item or "").strip().lower())
    if not normalized:
        return True
    generic_notes = (
        "use public transport",
        "public transport",
        "relaxed pace",
        "one cultural stop",
        "start from saldanha",
        "starting from saldanha",
        "plano com transportes públicos",
        "usar transportes públicos",
        "ritmo descontraído",
        "uma paragem cultural",
        "partida de saldanha",
    )
    return normalized in generic_notes


def _source_is_materially_used(source_id: str, rendered_body: str) -> bool:
    """Return whether a source supports facts actually present in the plan."""
    text = re.sub(r"\s+", " ", (rendered_body or "").lower())
    if not text:
        return True
    if source_id == "ipma":
        if any(marker in text for marker in ("weather was not confirmed", "no weather-specific", "tempo não confirmado", "tempo nao confirmado")):
            return False
        return any(marker in text for marker in ("weather", "rain", "temperature", "wind", "chuva", "temperatura", "vento", "☔"))
    if source_id == "metro":
        return any(marker in text for marker in ("metro", "yellow line", "blue line", "green line", "red line", "linha amarela", "linha azul", "linha verde", "linha vermelha"))
    if source_id == "carris":
        if any(marker in text for marker in ("carris line numbers and schedules are not needed", "carris line numbers and schedules should be confirmed")):
            return False
        if re.search(r"\bcarris\s+\d{1,4}[a-z]?\b", text):
            return True
        return any(
            marker in text
            for marker in (
                "carris",
                "bus",
                "tram",
                "autocarro",
                "elétrico",
                "eletrico",
                "opções carris",
                "opcoes carris",
                "pç. figueira",
                "pç figueira",
                "pç. comércio",
                "estação fluvial belém",
            )
        )
    if source_id == "carris_metropolitana":
        return any(marker in text for marker in ("carris metropolitana", "suburban bus", "autocarro suburbano"))
    if source_id == "cp":
        return any(marker in text for marker in (" cp ", "comboio", "train", "cascais line", "linha de cascais"))
    if source_id.startswith("visitlisboa"):
        return any(marker in text for marker in ("visitlisboa", "museum", "museu", "cultural", "culture", "restaurant", "restaurante", "event", "evento", "príncipe real", "principe real", "cam "))
    if source_id == "lisboa_aberta":
        return any(
            marker in text
            for marker in (
                "lisboa aberta", "dados abertos", "municipal",
                "pharmacy", "farmácia", "hospital", "biblioteca", "library",
                "escola", "school", "mercado", "market", "polícia", "police",
                "bombeiros", "firefighters", "wc", "toilet", "restroom",
                "parque infantil", "playground",
            )
        )
    return True


def _clean_inline(value: str) -> str:
    """Clean one inline value before rendering it into Markdown."""
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^#{1,6}\s*", "", text).strip()
    text = re.sub(r"\*\*([^*:]{2,80}):\s+\*\*", r"**\1:**", text)
    text = re.sub(r"\*\*([^*:]{2,80}):\s*\*\*", r"**\1:**", text)
    text = re.sub(
        r"\b(Transfer at|Continue on|Exit at|Start at|Route|Estimated total time|Nearest metro to [^:]{1,80}|Mudar em|Continuar em|Sair em|Começar em|Comecar em|Rota|Percurso|Tempo estimado):(?=[^\s*])",
        r"\1: ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"[`{}]", "", text)
    return text.strip(" -–—")[:260]


def _normalize_for_match(value: str) -> str:
    """Normalize a short value for duplicate and semantic matching."""
    text = str(value or "").lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[\*_`#\[\]().,:;!?|/\\-]+", " ", text)
    text = re.sub(r"[^\wÀ-ÿ\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_list(items: Iterable[str], max_items: int = 5) -> List[str]:
    """Clean, deduplicate, and limit a list of Markdown bullet values."""
    output: List[str] = []
    forbidden = {"", "n/a", "na", "none", "null", "unknown", "not available", "not provided", "+ info"}
    for item in items or []:
        text = _clean_inline(str(item or ""))
        if not text or text.lower().strip(" .:-_") in forbidden:
            continue
        if re.fullmatch(
            r"[\U0001F300-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\uFE0F\u200D\s]+(?:\*\*[A-Za-zÀ-ÿ0-9 /'-]{2,70}:\*\*)?",
            text,
        ):
            continue
        if re.fullmatch(r"\*\*[A-Za-zÀ-ÿ0-9 /'-]{2,70}:\*\*", text):
            continue
        if text not in output:
            output.append(text)
        if len(output) >= max_items:
            break
    return output


def _clean_markdown(markdown: str) -> str:
    """Apply final Markdown cleanup before returning the rendered plan."""
    markdown = re.sub(r"\n{3,}", "\n\n", markdown.strip())
    markdown = re.sub(r"(?m)^###\s+([^*\n]+)$", lambda m: f"### **{m.group(1).strip()}**", markdown)
    markdown = re.sub(r"(?m)^\s*[-*•]\s*$\n?", "", markdown)
    return markdown.strip()
