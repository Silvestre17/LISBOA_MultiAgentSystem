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
from datetime import datetime
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

    constraint_items = _clean_list(draft.constraints_used, max_items=6) or [
        "plano pedido pelo utilizador" if is_pt else "the user's requested plan"
    ]
    lines.extend(["", "---", "", f"### 🧭 **{'Restrições usadas' if is_pt else 'Constraints used'}**"])
    for item in constraint_items:
        lines.append(f"    - 🎯 **{'Critério' if is_pt else 'Criterion'}:** {item}")

    if draft.blocks:
        lines.extend(["", "---", "", f"### 📍 **{'Blocos do plano' if is_pt else 'Plan blocks'}**"])
        for index, block in enumerate(draft.blocks[:5], start=1):
            emoji = _BLOCK_KIND_EMOJI.get((block.kind or "").lower(), "📍")
            block_title = _clean_inline(block.title) or (f"Bloco {index}" if is_pt else f"Block {index}")
            block_title = re.sub(r"^(?:Block|Bloco)\s*\d+\s*[·:-]\s*", "", block_title, flags=re.IGNORECASE).strip()
            lines.extend(["", f"### {emoji} **{'Bloco' if is_pt else 'Block'} {index} · {block_title}**"])
            if block.purpose:
                lines.append(f"    - 🎯 **{'Objetivo' if is_pt else 'Purpose'}:** {_clean_inline(block.purpose)}")
            for detail in _clean_list(block.details, max_items=4):
                lines.append(f"    - 📝 **{'Detalhe' if is_pt else 'Detail'}:** {detail}")
            for movement in _clean_list(block.movement, max_items=3):
                lines.append(f"    - 🚇 **{'Movimento' if is_pt else 'Movement'}:** {movement}")
            for weather in _clean_list(block.weather, max_items=2):
                lines.append(f"    - ☔ **{'Ajuste meteorológico' if is_pt else 'Weather adjustment'}:** {weather}")
            for limitation in _clean_list(block.limitations, max_items=2):
                lines.append(f"    - ⚠️ **{'Limite' if is_pt else 'Limit'}:** {limitation}")

    movement_items = _clean_list(draft.movement_logic, max_items=6) or [
        "seguir a ordem indicada e confirmar pernas exatas no operador quando não estiverem evidenciadas"
        if is_pt
        else "follow the order shown and confirm exact legs with the operator when they are not evidenced"
    ]
    weather_items = _clean_list(draft.weather_strategy, max_items=6) or [
        "sem adaptação meteorológica adicional para além dos dados evidenciados"
        if is_pt
        else "no additional weather adaptation beyond the evidenced data"
    ]
    limitation_items = _clean_list(draft.limitations, max_items=6) or [
        "horários, bilhetes, preços, reservas e disponibilidade em tempo real só estão confirmados quando indicados acima"
        if is_pt
        else "opening hours, tickets, prices, bookings, and live availability are confirmed only where stated above"
    ]

    _append_section(lines, "🚇", "Lógica de movimento" if is_pt else "Movement logic", movement_items, "Transporte" if is_pt else "Movement")
    _append_section(lines, "☔", "Estratégia meteorológica" if is_pt else "Weather strategy", weather_items, "Tempo" if is_pt else "Weather")
    _append_section(lines, "💡", "Dicas" if is_pt else "Tips", draft.tips, "Dica" if is_pt else "Tip")
    _append_section(lines, "⚠️", "Limitações" if is_pt else "Limitations", limitation_items, "Limite" if is_pt else "Limit")

    footer = _source_footer(draft, sources, is_pt)
    if footer:
        lines.extend(["", footer])
    return _clean_markdown("\n".join(lines))


def _append_section(lines: List[str], emoji: str, title: str, items: Iterable[str], label: str) -> None:
    """Append a labeled Markdown section when it has valid items.

    Args:
        lines: Mutable Markdown line buffer.
        emoji: Section emoji used by the response contract.
        title: Section heading text.
        items: Candidate bullet items.
        label: Per-item label used before each bullet value.
    """
    cleaned = _clean_list(list(items), max_items=6)
    if not cleaned:
        return
    lines.extend(["", "---", "", f"### {emoji} **{title}**"])
    for item in cleaned:
        lines.append(f"    - {emoji} **{label}:** {item}")


def _source_footer(draft: PlanDraft, sources: Dict[str, SourceRef], is_pt: bool) -> str:
    """Build a source footer from source identifiers materially used in a plan.

    Args:
        draft: Structured plan whose source identifiers are inspected.
        sources: Public source references keyed by source identifier.
        is_pt: Whether the rendered answer is in Portuguese.

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
    if not deduped:
        return ""
    links = []
    for source_id in deduped[:5]:
        source = sources[source_id]
        label = source.label_pt if is_pt else source.label_en
        links.append(f"[*{label}*]({source.url})")
    timestamp = datetime.now().strftime("%H:%M")
    return f"📌 **{'Fonte' if is_pt else 'Source'}:** {' | '.join(links)} | **{'Atualizado' if is_pt else 'Updated'}:** {timestamp}"


def _clean_inline(value: str) -> str:
    """Clean one inline value before rendering it into Markdown."""
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[`{}\[\]]", "", text)
    return text.strip(" -–—")[:260]


def _clean_list(items: Iterable[str], max_items: int = 5) -> List[str]:
    """Clean, deduplicate, and limit a list of Markdown bullet values."""
    output: List[str] = []
    forbidden = {"", "n/a", "na", "none", "null", "unknown", "not available", "not provided", "+ info"}
    for item in items or []:
        text = _clean_inline(str(item or ""))
        if not text or text.lower().strip(" .:-_") in forbidden:
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
