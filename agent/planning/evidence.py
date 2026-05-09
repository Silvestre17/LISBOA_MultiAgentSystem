# ==========================================================================
# Master Thesis - Planner Evidence Extraction
#   - André Filipe Gomes Silvestre, 20240502
#
#   Converts grounded worker-agent Markdown into compact evidence cards for
#   PlannerAgent. The extractor is intentionally conservative: it preserves
#   visible facts already emitted by weather, transport, researcher, or event
#   workers, and it never creates new venues, schedules, prices, routes, or
#   weather claims.
# ==========================================================================

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from agent.planning.models import EvidenceCard, SourceRef


SOURCE_CATALOG: Dict[str, SourceRef] = {
    "ipma": SourceRef("ipma", "IPMA", "IPMA", "https://www.ipma.pt"),
    "visitlisboa_places": SourceRef("visitlisboa_places", "VisitLisboa Places", "VisitLisboa Locais", "https://www.visitlisboa.com/en/places"),
    "visitlisboa_events": SourceRef("visitlisboa_events", "VisitLisboa Events", "VisitLisboa Eventos", "https://www.visitlisboa.com/en/events"),
    "metro": SourceRef("metro", "Metro de Lisboa", "Metro de Lisboa", "https://www.metrolisboa.pt"),
    "carris": SourceRef("carris", "Carris", "Carris", "https://www.carris.pt"),
    "carris_metropolitana": SourceRef("carris_metropolitana", "Carris Metropolitana", "Carris Metropolitana", "https://www.carrismetropolitana.pt"),
    "cp": SourceRef("cp", "CP", "CP", "https://www.cp.pt"),
    "lisboa_aberta": SourceRef("lisboa_aberta", "Lisboa Aberta", "Lisboa Aberta", "https://dados.cm-lisboa.pt/"),
}


@dataclass
class EvidenceBundle:
    """Structured evidence available to a single planning turn.

    Attributes:
        cards: Evidence cards extracted from worker-agent outputs.
        sources: Source references keyed by source identifier.
        limitations: Constraints or caveats that the planner must preserve.
    """

    cards: List[EvidenceCard] = field(default_factory=list)
    sources: Dict[str, SourceRef] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)

    def cards_by_kind(self, *kinds: str) -> List[EvidenceCard]:
        """Return cards whose kind matches one of the requested kinds.

        Args:
            *kinds: Card kinds to include, compared case-insensitively.

        Returns:
            Matching evidence cards in their original order.
        """
        allowed = {kind.lower() for kind in kinds}
        return [card for card in self.cards if card.kind.lower() in allowed]

    def source_ids(self) -> List[str]:
        """Return unique source identifiers used by the bundle cards.

        Returns:
            Source identifiers that are present both in cards and in the bundle
            source registry.
        """
        ids: List[str] = []
        for card in self.cards:
            ids.extend(card.source_ids)
        return list(dict.fromkeys(source_id for source_id in ids if source_id in self.sources))

    def to_prompt_text(self, max_cards: int = 22) -> str:
        """Render evidence cards as compact text for the planner prompt.

        Args:
            max_cards: Maximum number of card entries to include per section.

        Returns:
            Prompt-ready evidence text, including limitations when available.
        """
        if not self.cards:
            return "No structured evidence cards were extracted. Do not invent venues, routes, event dates, prices, or weather facts."
        sections: List[str] = []
        for kind in ("weather", "transport", "place", "event", "service", "knowledge"):
            cards = [card for card in self.cards if card.kind == kind]
            if not cards:
                continue
            sections.append(f"## {kind.upper()} EVIDENCE")
            for card in cards[:max_cards]:
                sections.append(card.to_prompt_text())
        if self.limitations:
            sections.append("## LIMITATIONS")
            sections.extend(f"- {item}" for item in self.limitations[:8])
        return "\n\n".join(sections[: max_cards * 2])


def normalize_text(text: str) -> str:
    """Normalize text for loose matching and duplicate detection.

    Args:
        text: Raw text to normalize.

    Returns:
        Lowercase ASCII-like text with punctuation collapsed to spaces.
    """
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"[^a-zA-Z0-9\s/-]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip().lower()


def build_evidence_bundle(
    *,
    weather_data: str = "",
    transport_data: str = "",
    places_data: str = "",
    events_data: str = "",
    qa_disclaimers: Sequence[str] | None = None,
) -> EvidenceBundle:
    """Build a structured evidence bundle from worker-agent outputs.

    Args:
        weather_data: WeatherAgent Markdown or text output.
        transport_data: TransportAgent Markdown or text output.
        places_data: ResearcherAgent place or service output.
        events_data: ResearcherAgent event output.
        qa_disclaimers: Optional QA caveats to carry into planning.

    Returns:
        Evidence cards, source references, and limitations for PlannerAgent.
    """
    cards: List[EvidenceCard] = []
    sources: Dict[str, SourceRef] = {}

    def extend(new_cards: Iterable[EvidenceCard]) -> None:
        """Append valid cards and register only supported source identifiers."""
        for card in new_cards:
            if not card.title and not card.summary:
                continue
            card.source_ids = [source_id for source_id in dict.fromkeys(card.source_ids) if source_id in SOURCE_CATALOG]
            for source_id in card.source_ids:
                sources[source_id] = SOURCE_CATALOG[source_id]
            cards.append(card)

    extend(_extract_weather_cards(weather_data))
    extend(_extract_transport_cards(transport_data))
    extend(_extract_research_cards(places_data, default_kind="place"))
    extend(_extract_research_cards(events_data, default_kind="event"))

    limitations = [str(item).strip() for item in (qa_disclaimers or []) if str(item).strip()]
    limitations.extend(_extract_limitations("\n".join([weather_data, transport_data, places_data, events_data])))
    return EvidenceBundle(cards=_dedupe_cards(cards), sources=sources, limitations=list(dict.fromkeys(limitations)))


def _detect_sources(text: str, default: Sequence[str] = ()) -> List[str]:
    """Infer known public sources mentioned in extracted text.

    Args:
        text: Worker output or evidence fragment to inspect.
        default: Source identifiers to include before text-based detection.

    Returns:
        Deduplicated source identifiers present in ``SOURCE_CATALOG``.
    """
    lowered = (text or "").lower()
    sources = list(default)
    if "ipma" in lowered or "weather" in lowered or "meteorolog" in lowered:
        sources.append("ipma")
    if "visitlisboa.com" in lowered or "visitlisboa" in lowered:
        if "/events" in lowered or "event" in lowered or "evento" in lowered:
            sources.append("visitlisboa_events")
        else:
            sources.append("visitlisboa_places")
    if "metrolisboa" in lowered or "metro de lisboa" in lowered:
        sources.append("metro")
    if re.search(r"\bmetro\b", lowered) or re.search(
        r"\b(?:yellow|blue|green|red|amarela|azul|verde|vermelha)\s+line\b|\blinha\s+(?:amarela|azul|verde|vermelha)\b",
        lowered,
    ):
        sources.append("metro")
    if "carrismetropolitana" in lowered or "carris metropolitana" in lowered:
        sources.append("carris_metropolitana")
    if "carris.pt" in lowered or re.search(r"\bcarris\b", lowered):
        sources.append("carris")
    if "cp.pt" in lowered or re.search(r"\bcp\b|comboios de portugal", lowered):
        sources.append("cp")
    if "dados.cm-lisboa" in lowered or "lisboa aberta" in lowered:
        sources.append("lisboa_aberta")
    return [source for source in dict.fromkeys(sources) if source in SOURCE_CATALOG]


def _visible_line(line: str) -> str:
    """Remove Markdown-only decoration while preserving visible content."""
    line = re.sub(r"\[[^\]]+\]\(([^)]+)\)", lambda match: match.group(0), line or "")
    line = re.sub(r"^[\s\-*•]+", "", line.strip())
    line = re.sub(r"#{1,6}\s*", "", line).strip()
    line = re.sub(r"\*\*", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def _clean_title(raw: str) -> str:
    """Normalize a candidate card title for prompt-safe evidence use."""
    title = _visible_line(raw)
    title = re.sub(r"^[\W_]+", "", title).strip(" :-–—·")
    title = re.sub(r"^(?:Block|Bloco|Day|Dia)\s*\d+\s*[·:-]\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title)
    if len(title) > 96:
        title = title[:93].rstrip() + "..."
    return title


def _extract_weather_cards(text: str) -> List[EvidenceCard]:
    """Extract one compact weather evidence card from worker output.

    Args:
        text: Weather worker response text.

    Returns:
        A list containing a weather card when usable content is present.
    """
    if not text:
        return []
    sources = _detect_sources(text, default=["ipma"])
    lines = [_visible_line(line) for line in text.splitlines() if _visible_line(line)]
    useful: List[str] = []
    for line in lines:
        lowered = normalize_text(line)
        if lowered.startswith(("source", "fonte", "updated", "atualizado")) or set(line) == {"="}:
            continue
        if any(token in lowered for token in ("warning", "aviso", "temperature", "temperatura", "rain", "chuva", "wind", "vento", "conditions", "condicoes", "condições", "forecast", "previsao", "previsão")):
            useful.append(line)
    if not useful and lines:
        useful = lines[:5]
    return [
        EvidenceCard(
            id="weather_1",
            kind="weather",
            title="Weather conditions for Lisbon",
            summary="; ".join(useful[:5]),
            source_ids=sources,
        )
    ]


def _extract_transport_cards(text: str) -> List[EvidenceCard]:
    """Extract one compact transport evidence card from worker output.

    Args:
        text: Transport worker response text.

    Returns:
        A list containing a transport card when usable content is present.
    """
    if not text:
        return []
    sources = _detect_sources(text)
    lines = [_visible_line(line) for line in text.splitlines() if _visible_line(line)]
    useful: List[str] = []
    for line in lines:
        lowered = normalize_text(line)
        if lowered.startswith(("source", "fonte", "updated", "atualizado")) or set(line) == {"="}:
            continue
        if any(token in lowered for token in ("route", "rota", "line", "linha", "metro", "carris", "cp", "train", "bus", "tram", "departure", "partida", "board", "embar", "alight", "exit", "transfer", "real time", "tempo real")):
            useful.append(line)
    if not useful and lines:
        useful = lines[:6]
    return [
        EvidenceCard(
            id="transport_1",
            kind="transport",
            title="Transport evidence",
            summary="; ".join(useful[:7]),
            source_ids=sources,
        )
    ]


def _extract_research_cards(text: str, *, default_kind: str) -> List[EvidenceCard]:
    """Extract place, event, service, or knowledge cards from research output.

    Args:
        text: Researcher worker response text.
        default_kind: Kind to use when the content does not imply a narrower
            evidence type.

    Returns:
        Evidence cards extracted from recognizable Markdown sections, or a
        fallback summary card when only plain text is available.
    """
    if not text:
        return []
    sources = _detect_sources(text)
    sections = _split_markdown_cards(text)
    cards: List[EvidenceCard] = []
    for index, section in enumerate(sections[:10], start=1):
        title = _clean_title(section[0]) if section else ""
        if _is_non_card_title(title):
            continue
        fields = _extract_fields(section[1:])
        summary_parts = []
        for key in (
            "Description",
            "Descrição",
            "Category",
            "Categoria",
            "When",
            "Quando",
            "Venue",
            "Local",
            "Address",
            "Morada",
            "Price",
            "Preço",
            "Hours",
            "Horário",
            "Horários",
            "Website",
            "More details",
            "Mais detalhes",
            "Tickets",
            "Bilhetes",
        ):
            if fields.get(key):
                summary_parts.append(f"{key}: {fields[key]}")
        if not summary_parts:
            for line in section[1:4]:
                visible = _visible_line(line)
                if visible and not _is_noise_line(visible):
                    summary_parts.append(visible)
        kind = _infer_card_kind(title, fields, default_kind)
        card_sources = _detect_sources("\n".join(section), default=sources or (["visitlisboa_events"] if kind == "event" else ["visitlisboa_places"]))
        cards.append(
            EvidenceCard(
                id=f"{kind}_{index}",
                kind=kind,
                title=title,
                summary="; ".join(summary_parts[:5]),
                fields=fields,
                source_ids=card_sources,
            )
        )
    if not cards:
        useful = [_visible_line(line) for line in text.splitlines() if _visible_line(line) and not _is_noise_line(_visible_line(line))]
        if useful:
            cards.append(
                EvidenceCard(
                    id=f"{default_kind}_1",
                    kind=default_kind,
                    title=useful[0][:96],
                    summary="; ".join(useful[1:6]),
                    source_ids=sources,
                )
            )
    return cards


def _split_markdown_cards(text: str) -> List[List[str]]:
    """Split Markdown output into card-like sections.

    Args:
        text: Markdown text emitted by a worker.

    Returns:
        Candidate card sections, each represented as a list of source lines.
    """
    lines = [line.rstrip() for line in text.splitlines()]
    cards: List[List[str]] = []
    current: List[str] = []
    header_re = re.compile(r"^\s*(?:###\s+|[-*•]\s+)?(?:[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s*)?\*\*[^*]{2,120}\*\*\s*$")
    alt_header_re = re.compile(r"^\s*###\s+")
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped == "---":
            continue
        looks_header = bool(header_re.match(stripped) or alt_header_re.match(stripped)) and not re.search(r"\*\*(?:Source|Fonte|Updated|Atualizado):\*\*", stripped, re.IGNORECASE)
        if looks_header:
            if current:
                cards.append(current)
            current = [stripped]
        elif current:
            current.append(stripped)
    if current:
        cards.append(current)
    return cards


def _extract_fields(lines: Sequence[str]) -> Dict[str, str]:
    """Extract labeled Markdown fields from card body lines.

    Args:
        lines: Lines below a card heading.

    Returns:
        Field labels mapped to cleaned values, excluding placeholders.
    """
    fields: Dict[str, str] = {}
    field_re = re.compile(r"^\s*[-*•]?\s*(?:[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s*)?\*\*(?P<label>[^*:]{2,40}):?\*\*\s*:??\s*(?P<value>.+)$")
    plain_field_re = re.compile(
        r"^\s*[-*•]?\s*(?:[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s*)?"
        r"(?P<label>Description|Descrição|Descricao|Category|Categoria|When|Quando|Venue|Local|Location|Address|Morada|Price|Preço|Preco|Hours|Horário|Horario|Horários|Horarios|Today|Hoje|Website|URL|More details|Mais detalhes|Tickets|Bilhetes)\s*:\s*(?P<value>.+)$",
        flags=re.IGNORECASE,
    )
    raw_url_re = re.compile(r"^\s*[-*•]?\s*(?:🔗\s*)?(?P<url>https?://\S+)\s*$", flags=re.IGNORECASE)
    for raw in lines:
        stripped = raw.strip()
        match = field_re.match(stripped) or plain_field_re.match(stripped)
        if match:
            label = _canonical_field_label(_visible_line(match.group("label")).strip(" :"))
            value = _visible_line(match.group("value"))
        else:
            url_match = raw_url_re.match(stripped)
            if not url_match:
                continue
            label = "Website"
            value = url_match.group("url").rstrip(").,;")
        if _is_metadata_field_label(label):
            continue
        if not label or not value or _is_missing(value):
            continue
        if label not in fields:
            fields[label] = value[:240]
    return fields


def _canonical_field_label(label: str) -> str:
    """Normalize equivalent user-facing field labels for planner evidence.

    Args:
        label: Raw label extracted from a card line.

    Returns:
        Canonical label used by the planner prompt and renderer.
    """
    normalized = normalize_text(label)
    mapping = {
        "descricao": "Description",
        "description": "Description",
        "category": "Category",
        "categoria": "Category",
        "when": "When",
        "quando": "When",
        "venue": "Venue",
        "local": "Venue",
        "location": "Venue",
        "address": "Address",
        "morada": "Address",
        "price": "Price",
        "preco": "Price",
        "hours": "Hours",
        "horario": "Hours",
        "horarios": "Hours",
        "today": "Hours",
        "hoje": "Hours",
        "website": "Website",
        "url": "Website",
        "more details": "Website",
        "mais detalhes": "Website",
        "tickets": "Tickets",
        "bilhetes": "Tickets",
    }
    return mapping.get(normalized, label.strip())


def _is_metadata_field_label(label: str) -> bool:
    """Return whether a parsed label is result metadata rather than card evidence."""
    normalized = normalize_text(label)
    return normalized in {
        "filter used",
        "filtro aplicado",
        "result count",
        "contagem de resultados",
        "highlights shown",
        "destaques mostrados",
        "result window",
        "janela de resultados",
        "source mix",
        "mistura de fontes",
        "source completeness note",
        "nota de completude da fonte",
    }


def _infer_card_kind(title: str, fields: Dict[str, str], default_kind: str) -> str:
    """Infer the evidence card kind from a title and extracted fields."""
    text = normalize_text(" ".join([title, *fields.keys(), *fields.values()]))
    if any(token in text for token in ("when", "quando", "event", "evento", "exhibition", "festival", "fair", "feira")):
        return "event"
    if any(token in text for token in ("pharmacy", "hospital", "library", "parking", "ecoponto", "servico", "serviço")):
        return "service"
    return default_kind


def _extract_limitations(text: str) -> List[str]:
    """Extract explicit limitations that should constrain planning."""
    if not text:
        return []
    limitations: List[str] = []
    for line in text.splitlines():
        visible = _visible_line(line)
        lowered = normalize_text(visible)
        if any(token in lowered for token in ("not confirmed", "nao confirmado", "não confirmado", "unavailable", "indisponivel", "não disponível", "nao disponivel", "limitation", "limite")):
            limitations.append(visible)
    return limitations[:8]


def _is_missing(value: str) -> bool:
    """Return whether a field value is a placeholder rather than evidence."""
    normalized = normalize_text(value).strip(" .:-_")
    return normalized in {"", "n/a", "na", "unknown", "not available", "not provided", "indisponivel", "nao disponivel", "não disponível", "+ info"}


def _is_noise_line(value: str) -> bool:
    """Return whether a line is metadata or formatting noise."""
    normalized = normalize_text(value)
    return not normalized or normalized.startswith(("source", "fonte", "updated", "atualizado", "result window", "janela de resultados")) or set(value) == {"="}


def _is_non_card_title(title: str) -> bool:
    """Return whether a heading is structural text rather than an evidence card."""
    normalized = normalize_text(title)
    if not normalized:
        return True
    blocked = (
        "direct answer", "resposta direta", "constraints", "restricoes", "movement logic", "weather strategy", "limitations", "source", "fonte", "local highlights", "destaques locais", "summary", "resumo", "places and attractions", "locais e atracoes", "events found", "eventos encontrados",
    )
    return any(token in normalized for token in blocked)


def _dedupe_cards(cards: Sequence[EvidenceCard]) -> List[EvidenceCard]:
    """Remove duplicate evidence cards while preserving first occurrence order."""
    deduped: List[EvidenceCard] = []
    seen: set[tuple[str, str]] = set()
    for card in cards:
        key = (card.kind, normalize_text(card.title))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(card)
    return deduped
