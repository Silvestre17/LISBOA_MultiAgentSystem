# ==========================================================================
# Master Thesis - Brief-Driven Plan Composition Helpers
#   - André Filipe Gomes Silvestre, 20240502
#
#   Deterministic steps around the planner LLM for requests that come with a
#   planning brief: rebuild each stop's details from its evidence card, ask
#   the Transport worker for the legs between the chosen stops, and merge
#   those legs into the plan before rendering. The LLM decides which stops
#   and in which order; these helpers make sure the details and the legs
#   shown to the user are the ones the tools returned.
# ==========================================================================

from __future__ import annotations

import copy
import math
import re
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence

from agent.planning.brief import PlanBrief, plan_date
from agent.planning.evidence import SOURCE_CATALOG, EvidenceBundle, normalize_text
from agent.planning.legs import LegRequest, estimate_leg_minutes, parse_leg_endpoint
from agent.planning.models import PlanBlock, PlanDraft, SourceRef
from agent.planning.quality import (
    DEFAULT_STAY_MINUTES,
    card_coordinates,
    card_matches_kind,
    closed_all_day,
    hours_for_plan_day,
    match_block_to_card,
    open_at,
    strip_block_time,
    without_public_cards,
)
from tools.utils import haversine_distance, lisbon_now

LegResolver = Callable[..., List[str]]

_CONCRETE_LEG_RE = re.compile(
    r"\b(?:metro|linha|line|tram|electrico|eletrico|bus|autocarro|train|comboio|cp|carris|ferry|walk|a pe|caminh)\b"
)
_FOOD_CARD_RE = re.compile(r"\b(?:restaurant|restaurante|cafe|coffee|pastelaria|pastry|bar|tasca|taberna)\b")
# Notes about legs or connections: code resolves the legs and adds its own
# note when one cannot be confirmed, so model notes on them are stale.
_CONNECTION_WORD_RE = re.compile(
    r"\b(?:connection|connections|leg|legs|route|routes|transit|transport|transfer|walking|walk|tram|bus|metro|train|"
    r"ligacao|ligacoes|percurso|trajeto|deslocacao|deslocacoes|transportes?|eletrico|autocarro|comboio|a pe)\b"
)
_NOT_CONFIRMED_RE = re.compile(
    r"\b(?:not confirmed|unconfirmed|could not be confirmed|confirmed only|only confirmed|only .{0,60} (?:is|are|was|were) confirmed|"
    r"nao (?:foi |foram |ficou |ficaram |esta |estao )?confirmad\w*|por confirmar|sem confirmacao|"
    r"not available|unavailable|nao (?:esta|estava|estao|estavam) disponive\w*|indisponive\w*|"
    r"no confirmed|sem (?:ligacao|conexao|percurso) confirmad\w*)\b"
)
_INTERNAL_NOTE_RE = re.compile(
    r"(?<!lisboa )\bcards?\b|(?<!lisboa )\bcart(?:ao|oes)\b|\bevidence\b|\bevidencias?\b|\bgathered data\b|\bdados recolhidos\b|"
    r"\bfound for\b|\bencontrado para\b|\btools?\b|\bagents?\b|\bi did not\b|\bnao publiquei\b|\bfichas?\b|"
    r"\bpedido revisto\b|\brevisao pedida\b|\b(?:revised|updated) (?:request|plan|schedule|timing)\b|"
    r"\b(?:horario|plano) revisto\b|\bthe user\b|\bo utilizador\b|\ba utilizadora\b"
)
# Live departures and service-hours notices only hold for leaving now.
_LIVE_INFO_RES = (
    re.compile(r"\bnext\b.{0,40}\b(?:departures?|trains?|bus(?:es)?|trams?|metro)\b.{0,60}\b\d{1,2}:\d{2}", re.IGNORECASE),
    re.compile(r"\b(?:leaves|departs)\s+at\s+\d{1,2}:\d{2}", re.IGNORECASE),
    re.compile(r"\bpr[oó]xim[oa]s?\b.{0,40}\b(?:partidas?|comboios?|autocarros?|el[eé]tricos?|metros?)\b.{0,60}\b\d{1,2}[:h]\d{2}", re.IGNORECASE),
    re.compile(r"\bsai(?:em)?\s+(?:às|as)\s+\d{1,2}[:h]\d{2}", re.IGNORECASE),
    # "com partida às 15:41", "departure at 15:41", "departing at 15:41".
    re.compile(r"\b(?:partidas?|departures?|departing)\b[^.;,]{0,20}?\b(?:às|as|at)\s+\d{1,2}[:h]\d{2}", re.IGNORECASE),
    re.compile(r"\b(?:waiting time|wait time|tempo de espera)\b", re.IGNORECASE),
    re.compile(r"\b(?:outside (?:the )?(?:regular |normal )?(?:operating |service )?hours|fora do hor[aá]rio)\b", re.IGNORECASE),
    re.compile(r"\b(?:opera[cç][aã]o especial|special (?:operation|service)|para esta hora|at this hour|neste momento|right now)\b", re.IGNORECASE),
    # Current delays ("since some trains are delayed", "há atrasos").
    re.compile(
        r"\b(?:(?:are|is)\s+(?:currently\s+|now\s+)?(?:running\s+)?(?:delayed|late)|"
        r"(?:h[aá]|com|existem)\s+(?:alguns\s+)?atrasos|est[aã]o?\s+(?:com\s+)?atras(?:ad[oa]s?|os?))\b",
        re.IGNORECASE,
    ),
)
_TRACKING_PARAM_RE = re.compile(r"(?:utm_[a-z]+|fbclid|gclid)=[^&)\s]*", re.IGNORECASE)
_URL_QUERY_RE = re.compile(r"(https?://[^\s)?]+)\?([^\s)#]*)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[;,])\s+")
_BLOCK_TIME_RE = re.compile(r"^\s*(\d{1,2})[:h](\d{2})")
_LISBOA_CARD_RE = re.compile(r"\blisboa card\b", re.IGNORECASE)
_PRICE_VALUE_RE = re.compile(r"\d|\b(?:free|gr[aá]tis|gratuit[oa]|entrada livre)\b", re.IGNORECASE)


def plan_is_for_another_day(brief: Optional[PlanBrief]) -> bool:
    """Return whether the plan's first day is not today (a weekday that is today counts as today)."""
    if not brief or not brief.day:
        return False
    today = lisbon_now().date()
    return plan_date(brief, today) != today


def _hours_detail(value: str, *, future: bool, language: str) -> str:
    """Return the Hours detail, marking today's hours when the plan is for another day.

    Hours that name a weekday ("Sunday: 09:00 - 18:00") come from a search for
    the plan's day and need no note.
    """
    text = re.sub(r"^(?:Today|Hoje)\s*:\s*", "", value.strip(), flags=re.IGNORECASE)
    if not future or re.match(r"^[A-Za-zÀ-ú]+(?:-feira)?\s*:", text):
        return text
    return f"{text} (horário de hoje; confirma para o dia do plano)" if language == "pt" else f"{text} (today's hours; confirm for the day of the plan)"


def curate_block_details(draft: PlanDraft, evidence: EvidenceBundle, brief: Optional[PlanBrief], language: str) -> PlanDraft:
    """Rebuild each stop's details from its evidence card.

    The model's copied details are replaced by the card values, so an address,
    hours line, price, or link can never drift from the source. Stops that do
    not match any card are dropped.

    Args:
        draft: Plan returned by the planner LLM.
        evidence: Evidence cards available to the planner.
        brief: Planning brief, used to mark today's hours on future plans.
        language: Output language.

    Returns:
        The same draft, with curated blocks.
    """
    future = plan_is_for_another_day(brief)
    kept = []
    for block in draft.blocks:
        card = match_block_to_card(block.title, evidence)
        if card is None:
            continue
        fields: Dict[str, str] = getattr(card, "fields", {}) or {}
        time_prefix = block.title[: len(block.title) - len(strip_block_time(block.title))]
        block.title = f"{time_prefix}{card.title}".strip()
        details: List[str] = []
        if fields.get("Address"):
            details.append(f"Address: {fields['Address']}")
        elif fields.get("Venue"):
            details.append(f"Venue: {fields['Venue']}")
        elif fields.get("Coordinates"):
            # A place grounded from Wikipedia has coordinates but no street address.
            point = re.sub(r"\s+", "", str(fields["Coordinates"]))
            map_label = "Ver no mapa" if language == "pt" else "View on map"
            details.append(f"Location: [{map_label}](https://www.google.com/maps/search/?api=1&query={point})")
        if fields.get("When"):
            details.append(f"When: {fields['When']}")
        if fields.get("Hours"):
            hours_value = fields["Hours"]
            if " · " in hours_value and brief is not None:
                # Several plan days ("Saturday: ... · Sunday: ..."): the stop's own day.
                today = lisbon_now().date()
                block_day = plan_date(brief, today) + timedelta(days=max(0, int(getattr(block, "day", 0) or 1) - 1))
                hours_value = hours_for_plan_day(hours_value, block_day, today) or hours_value
            details.append(f"Hours: {_hours_detail(hours_value, future=future, language=language)}")
        price = str(fields.get("Price") or "").strip()
        if price and _LISBOA_CARD_RE.search(price):
            details.append(f"Lisboa Card: {_lisboa_card_benefit(price, language)}")
        elif price and _PRICE_VALUE_RE.search(price):
            details.append(f"Price: {price}")
        if fields.get("Tickets"):
            details.append(f"Tickets: {_strip_tracking(fields['Tickets'])}")
        if fields.get("Nearest metro") and brief is not None and brief.transport_mode != "walking":
            details.append(f"Nearest metro: {fields['Nearest metro']}")
        is_food = bool(_FOOD_CARD_RE.search(normalize_text(" ".join([card.title, fields.get("Category", ""), fields.get("Found for", "")]))))
        if is_food and fields.get("Features"):
            details.append(f"Features: {fields['Features']}")
        link_label = "More details" if fields.get("More details") else "Website" if fields.get("Website") else ""
        if link_label:
            details.append(f"{link_label}: {_strip_tracking(fields[link_label])}")
        block.details = details
        block.source_ids = list(dict.fromkeys([*block.source_ids, *card.source_ids]))
        kept.append(block)
    draft.blocks = kept
    return draft


def _strip_tracking(value: str) -> str:
    """Remove tracking parameters (utm_*, fbclid, gclid) from links in a field."""
    def _clean_query(match: re.Match) -> str:
        separator = "&amp;" if "&amp;" in match.group(2) else "&"
        kept = [
            part for part in match.group(2).split(separator)
            if part and not _TRACKING_PARAM_RE.fullmatch(part)
        ]
        return match.group(1) + ("?" + separator.join(kept) if kept else "")

    return _URL_QUERY_RE.sub(_clean_query, str(value or ""))


def _lisboa_card_benefit(price: str, language: str) -> str:
    """Rewrite a "10% with Lisboa Card" price field as the card benefit it is."""
    percent = re.search(r"(\d{1,3})\s*%", price)
    if percent:
        return f"{percent.group(1)}% de desconto" if language == "pt" else f"{percent.group(1)}% discount"
    if re.search(r"\b(?:free|gr[aá]tis|gratuit[oa])\b", price, re.IGNORECASE):
        return "entrada gratuita" if language == "pt" else "free entry"
    return re.sub(r"\s*(?:with|com)\s+Lisboa Card\b", "", price, flags=re.IGNORECASE).strip() or price


def _leg_query(block: Any, evidence: EvidenceBundle) -> tuple[str, str, Optional[tuple[float, float]]]:
    """Return the label, the lookup query (address or name), and stored coordinates of a stop."""
    card = match_block_to_card(block.title, evidence)
    label = strip_block_time(block.title)
    fields = (getattr(card, "fields", {}) or {}) if card is not None else {}
    address = parse_leg_endpoint(fields.get("Address") or fields.get("Venue") or "")
    return label, address or label, card_coordinates(card)


def _blocks_by_day(draft: PlanDraft) -> List[List[PlanBlock]]:
    """Group consecutive blocks by their day number."""
    groups: List[List[PlanBlock]] = []
    current_day: Optional[int] = None
    for block in draft.blocks:
        day = getattr(block, "day", 0)
        if not groups or day != current_day:
            groups.append([])
            current_day = day
        groups[-1].append(block)
    return groups


def _block_start_minutes(block: PlanBlock) -> Optional[int]:
    """Return a block's start time in minutes after midnight."""
    match = _BLOCK_TIME_RE.match(block.title or "")
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def _set_block_start(block: PlanBlock, minutes: int) -> None:
    """Rewrite the HH:MM prefix of a block title."""
    clock = f"{minutes // 60:02d}:{minutes % 60:02d}"
    block.title = f"{clock} · {strip_block_time(block.title)}"


def _is_trivial_walk(line: str) -> bool:
    """Return whether a walking leg is too short to be worth showing (<= 150 m)."""
    match = re.search(r"\((\d+)\s*m\b", line or "")
    return bool(line.startswith("🚶") and match and int(match.group(1)) <= 150)


def align_schedule_with_legs(
    blocks: List[PlanBlock],
    leg_minutes: List[Optional[int]],
    evidence: EvidenceBundle,
    *,
    plan_day: date,
    today: date,
    end_minutes: Optional[int] = None,
    language: str,
) -> None:
    """Push stop times later when the stay plus the leg does not fit.

    A stop never moves earlier, and an event keeps its fixed start. After a
    stop moves, it is checked again: outside the hours its card gives for
    the plan's day, or after the end time the user gave, it gets a note. A
    stop that no longer ends before a following event gets a note too.

    Args:
        blocks: One day's blocks, in visiting order.
        leg_minutes: Travel minutes for each consecutive pair (None if unknown).
        evidence: Evidence cards, for the event and hours checks.
        plan_day: Calendar date of these blocks.
        today: Current Lisbon date (card hours without a weekday are today's).
        end_minutes: The user's end time in minutes after midnight, if given.
        language: Output language.
    """
    is_pt = language == "pt"
    for block in blocks:
        start = _block_start_minutes(block)
        if start is not None and start % 5:
            _set_block_start(block, int(round(start / 5.0) * 5) % (24 * 60))

    def add_note(block: PlanBlock, note: str) -> None:
        if note not in block.limitations:
            block.limitations.append(note)

    for index in range(1, len(blocks)):
        previous, current = blocks[index - 1], blocks[index]
        previous_start = _block_start_minutes(previous)
        current_start = _block_start_minutes(current)
        if previous_start is None or current_start is None:
            continue
        card = match_block_to_card(current.title, evidence)
        stay = previous.stay_minutes or DEFAULT_STAY_MINUTES.get(previous.kind, 45)
        travel = leg_minutes[index - 1] if index - 1 < len(leg_minutes) and leg_minutes[index - 1] else 0
        earliest = int(math.ceil((previous_start + stay + travel) / 5.0) * 5)
        if current_start >= earliest or earliest >= 24 * 60:
            continue
        if getattr(card, "kind", "") == "event" or current.kind == "event":
            # The event keeps its time, so the stop before it has to be shorter.
            event_name = strip_block_time(current.title)
            add_note(
                previous,
                f"Encurta esta paragem para chegares a tempo a {event_name}." if is_pt
                else f"Keep this stop short to reach {event_name} on time.",
            )
            continue
        _set_block_start(current, earliest)
        raw_hours = str((getattr(card, "fields", {}) or {}).get("Hours", "")) if card is not None else ""
        if open_at(hours_for_plan_day(raw_hours, plan_day, today), earliest) is False:
            add_note(
                current,
                "A esta hora pode já estar fechado; confirma o horário." if is_pt
                else "It may be closed at this time; check the hours.",
            )
        if end_minutes is not None and earliest >= end_minutes:
            end_clock = f"{end_minutes // 60:02d}:{end_minutes % 60:02d}"
            add_note(
                current,
                f"Com as deslocações, esta paragem começa depois das {end_clock} que indicaste; retira-a se não tiveres tempo."
                if is_pt
                else f"With the travel time, this stop starts after the {end_clock} you gave; skip it if you run out of time.",
            )


_TRAIN_LINE_RE = re.compile(r"\b(?:train|comboio|cp)\b")
_LEG_ARRIVAL_RE = re.compile(r"→\s*([^:*]{2,60}?)\s*:")


def _train_last_mile_line(
    model_first_line: str,
    day_groups: List[List[PlanBlock]],
    evidence: EvidenceBundle,
    lines: List[str],
    roles: List[tuple[str, int]],
    resolver: Optional[LegResolver],
    language: str,
) -> str:
    """Return the walk from the arrival station of a train leg to the first stop.

    The train leg comes from the model's reading of the CP evidence; the
    station it arrives at is usually not the first stop (Sintra station and
    the Moorish Castle are 2 km apart), so the walk is measured here.
    """
    origin_unresolved = any(role == "origin" and day == 0 and not line for (role, day), line in zip(roles, lines))
    if resolver is None or not origin_unresolved or not day_groups or not _TRAIN_LINE_RE.search(normalize_text(model_first_line)):
        return ""
    arrival = _LEG_ARRIVAL_RE.search(model_first_line or "")
    if not arrival:
        return ""
    station = arrival.group(1).strip()
    station_query = f"Estação de {station}" if language == "pt" else f"{station} station"
    label, query, coordinates = _leg_query(day_groups[0][0], evidence)
    request = LegRequest(station_query, station_query, label, query, destination_coords=coordinates, walk_only=True)
    try:
        resolved = list(resolver([request], language=language, mode="walking") or [])
    except Exception:
        return ""
    line = resolved[0] if resolved else ""
    distance = re.search(r"\((\d+(?:[.,]\d+)?)\s*(km|m)\b", line or "")
    if not line or not distance:
        return ""
    kilometres = float(distance.group(1).replace(",", ".")) / (1000.0 if distance.group(2) == "m" else 1.0)
    return line if 0.15 < kilometres <= 3.0 else ""


_TRIP_DEPARTURE_RE = re.compile(
    r"(?:partida|parte|sai|saída|leaves?|departure|departs?)\D{0,40}?\b(?P<hour>\d{1,2})[:h](?P<minute>\d{2})\b",
    re.IGNORECASE,
)
_TRIP_DURATION_RE = re.compile(r"(?:dura[cç][aã]o|duration|cerca de|about|~)\D{0,15}?(?P<minutes>\d{1,3})\s*min", re.IGNORECASE)


def _start_after_origin_trip(blocks: List[PlanBlock], origin_line: str, last_mile_line: str) -> None:
    """Move the first stop after the arrival of the trip that reaches it.

    "Rossio → Sintra: next departure at 11:41, about 40 min" plus a 23 min
    walk means the first visit cannot start at 11:40. The later stops are
    then pushed by the usual schedule alignment.

    Args:
        blocks: The first day's blocks, in visiting order.
        origin_line: The movement line from the start point (the model's
            reading of the transport evidence, or the resolver's line).
        last_mile_line: The walk from the arrival station to the first stop, if any.
    """
    if not blocks or blocks[0].kind == "event":
        return
    departure = _TRIP_DEPARTURE_RE.search(origin_line or "")
    duration = _TRIP_DURATION_RE.search(origin_line or "") if departure else None
    first_start = _block_start_minutes(blocks[0])
    if not departure or not duration or first_start is None:
        return
    arrival = int(departure.group("hour")) * 60 + int(departure.group("minute")) + int(duration.group("minutes"))
    arrival += (estimate_leg_minutes(last_mile_line) or 0) if last_mile_line else 0
    earliest = int(math.ceil(arrival / 5.0) * 5)
    if first_start < earliest < 24 * 60:
        _set_block_start(blocks[0], earliest)


def resolve_plan_legs(
    draft: PlanDraft,
    evidence: EvidenceBundle,
    brief: PlanBrief,
    resolver: Optional[LegResolver],
    language: str,
) -> PlanDraft:
    """Add the movement between the chosen stops, as confirmed by the Transport worker.

    For each day, the resolver is asked for the leg from the start point to
    the first stop and for every pair of consecutive stops (walking distance,
    Metro route with the walks to and from the stations, or a direct Carris
    tram or bus). When it cannot confirm the first leg (for example a train
    trip it does not route), the model's first line, written from the
    Transport worker's route evidence, is kept. Pairs of stops the resolver
    cannot confirm get one shared limitation instead of an invented line.

    Args:
        draft: Plan with curated blocks.
        evidence: Evidence cards available to the planner.
        brief: Planning brief.
        resolver: Callable provided by the orchestrator that turns
            ``LegRequest`` objects into movement lines, or ``None``.
        language: Output language.

    Returns:
        The same draft with its movement lines merged.
    """
    if resolver is None or not draft.blocks:
        return draft

    requests: List[LegRequest] = []
    roles: List[tuple[str, int]] = []
    day_groups = _blocks_by_day(draft)
    for day_index, blocks in enumerate(day_groups):
        stops = [_leg_query(block, evidence) for block in blocks]
        if brief.origin:
            requests.append(
                LegRequest(brief.origin, brief.origin, stops[0][0], stops[0][1], prefer_transit=True, destination_coords=stops[0][2])
            )
            roles.append(("origin", day_index))
        for start, end in zip(stops, stops[1:]):
            requests.append(LegRequest(start[0], start[1], end[0], end[1], origin_coords=start[2], destination_coords=end[2]))
            roles.append(("stop", day_index))
        if brief.return_to_origin and brief.origin:
            requests.append(
                LegRequest(stops[-1][0], stops[-1][1], brief.origin, brief.origin, prefer_transit=True, origin_coords=stops[-1][2])
            )
            roles.append(("return", day_index))
        elif brief.end_point and day_index == len(day_groups) - 1:
            # "... and back to Oriente by bus": the plan ends at the end point.
            requests.append(
                LegRequest(stops[-1][0], stops[-1][1], brief.end_point, brief.end_point, prefer_transit=True, origin_coords=stops[-1][2])
            )
            roles.append(("return", day_index))
    if not requests:
        # No start point and one stop: nothing to move between, so a model
        # line about an unconfirmed leg describes a leg that does not exist.
        draft.movement_logic = [
            line for line in draft.movement_logic if not _NOT_CONFIRMED_RE.search(normalize_text(line))
        ]
        return draft

    try:
        lines = list(resolver(requests, language=language, mode=brief.transport_mode) or [])
    except Exception:
        return draft
    lines += [""] * (len(requests) - len(lines))

    model_first_line = draft.movement_logic[0] if draft.movement_logic else ""
    model_first_is_concrete = bool(
        model_first_line
        and _CONCRETE_LEG_RE.search(normalize_text(model_first_line))
        and not _NOT_CONFIRMED_RE.search(normalize_text(model_first_line))
    )
    movement: List[str] = []
    unresolved_stop_leg = False
    unresolved_return = False
    stop_leg_minutes: Dict[int, List[Optional[int]]] = {}
    for (role, day_index), line in zip(roles, lines):
        if role == "stop":
            stop_leg_minutes.setdefault(day_index, []).append(estimate_leg_minutes(line) if line else None)
        if line:
            # The start point sits at the first stop: no leg to show.
            if role in {"origin", "return"} and _is_trivial_walk(line):
                continue
            movement.append(line)
        elif role == "origin" and day_index == 0 and model_first_line:
            movement.append(model_first_line)
        elif role == "stop":
            unresolved_stop_leg = True
        elif role == "return":
            unresolved_return = True
    if not any(lines) and model_first_is_concrete and model_first_line not in movement:
        movement.insert(0, model_first_line)
    last_mile = _train_last_mile_line(model_first_line, day_groups, evidence, lines, roles, resolver, language)
    if last_mile and model_first_line in movement:
        movement.insert(movement.index(model_first_line) + 1, last_mile)
    draft.movement_logic = movement

    today = lisbon_now().date()
    first_day = plan_date(brief, today)
    end_minutes = None
    if brief.end_time:
        hour, minute = (int(part) for part in brief.end_time.split(":"))
        end_minutes = hour * 60 + minute
    if day_groups and model_first_line in movement:
        _start_after_origin_trip(day_groups[0], model_first_line, last_mile)
    for day_index, blocks in enumerate(day_groups):
        day_number = max(1, int(getattr(blocks[0], "day", 0) or 1))
        align_schedule_with_legs(
            blocks,
            stop_leg_minutes.get(day_index, []),
            evidence,
            plan_day=first_day + timedelta(days=day_number - 1),
            today=today,
            end_minutes=end_minutes if len(day_groups) == 1 else None,
            language=language,
        )

    if unresolved_return:
        target = brief.origin if brief.return_to_origin else brief.end_point
        note = (
            f"O regresso a {target} não ficou confirmado nos dados de transporte; confirma-o antes de sair."
            if language == "pt"
            else f"The way back to {target} was not confirmed by the transport data; check it before leaving."
        )
        if note not in draft.limitations:
            draft.limitations.append(note)
    if unresolved_stop_leg:
        note = (
            "Algumas ligações entre paragens não foram confirmadas nos dados de transporte; confirma-as num mapa antes de sair."
            if language == "pt"
            else "Some connections between stops were not confirmed by the transport data; check them on a map before leaving."
        )
        if note not in draft.limitations:
            draft.limitations.append(note)
    return draft


def _sentence_case(text: str) -> str:
    """Capitalize the first letter and end the text with a full stop."""
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return ""
    for index, char in enumerate(value):
        if char.isalpha():
            value = value[:index] + char.upper() + value[index + 1:]
            break
    if not re.search(r"[.!?)]$", value):
        value += "."
    return value


def _is_connection_note(text: str) -> bool:
    """Return whether a note says a leg or connection was not confirmed."""
    normalized = normalize_text(text)
    return bool(_CONNECTION_WORD_RE.search(normalized) and _NOT_CONFIRMED_RE.search(normalized))


def _strip_live_info(text: str) -> str:
    """Drop the clauses of a text that carry live departures or service-hours notices.

    Only the clause goes ("..., the next listed departure leaves at 05:30; ...");
    the rest of the sentence and a leg's "A → B:" label stay.
    """
    kept_sentences: List[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(str(text or "").strip()):
        if not any(pattern.search(sentence) for pattern in _LIVE_INFO_RES):
            kept_sentences.append(sentence)
            continue
        label = re.match(r"^(?:[^\w*]+\s*)?\*\*[^*\n]*→[^*\n]*:\*\*\s*", sentence)
        body = sentence[label.end():] if label else sentence
        clauses = [
            clause for clause in _CLAUSE_SPLIT_RE.split(body)
            if clause and not any(pattern.search(clause) for pattern in _LIVE_INFO_RES)
        ]
        rebuilt = " ".join(clauses).strip().rstrip(",;")
        if rebuilt and not rebuilt.endswith((".", "!", "?")):
            rebuilt += "."
        rebuilt = rebuilt[:1].upper() + rebuilt[1:]
        if label:
            rebuilt = f"{label.group(0)}{rebuilt}".strip()
        if rebuilt and rebuilt != (label.group(0).strip() if label else ""):
            kept_sentences.append(rebuilt)
    return " ".join(kept_sentences).strip()


def plan_starts_now(draft: PlanDraft, brief: Optional[PlanBrief]) -> bool:
    """Return whether the plan starts within the next 45 minutes."""
    now = lisbon_now()
    if plan_date(brief, now.date()) != now.date():
        return False
    first = next((_block_start_minutes(block) for block in draft.blocks if _block_start_minutes(block) is not None), None)
    if first is None:
        return True
    now_minutes = now.hour * 60 + now.minute
    return -15 <= first - now_minutes <= 45


# English "stop" in Portuguese text ("até ao primeiro stop"), with the
# article made feminine for "paragem".
_PT_FEMININE_ARTICLES = {
    "o": "a", "ao": "à", "do": "da", "no": "na", "este": "esta", "esse": "essa",
    "os": "as", "aos": "às", "dos": "das", "nos": "nas", "estes": "estas", "esses": "essas",
}
_PT_STOP_RE = re.compile(
    r"\b(?:(?P<article>o|ao|do|no|este|esse|os|aos|dos|nos|estes|esses)\s+)?"
    r"(?:(?P<ordinal>primeiro|último|ultimo|segundo|próximo|proximo)\s+)?(?P<noun>stops?)\b",
    re.IGNORECASE,
)
_PT_FEMININE_ORDINALS = {"primeiro": "primeira", "último": "última", "ultimo": "última", "segundo": "segunda", "próximo": "próxima", "proximo": "próxima"}


def _portuguese_stop_word(text: str) -> str:
    """Replace the English word "stop" in Portuguese text with "paragem"."""

    def replace(match: re.Match) -> str:
        article, ordinal, noun = match.group("article"), match.group("ordinal"), match.group("noun")
        if not (article or ordinal):
            # A bare "stop" may be part of an English name ("Hop-on bus stop").
            return match.group(0)
        words = []
        if article:
            feminine = _PT_FEMININE_ARTICLES[article.lower()]
            words.append(feminine[0].upper() + feminine[1:] if article[0].isupper() else feminine)
        if ordinal:
            feminine = _PT_FEMININE_ORDINALS[ordinal.lower()]
            words.append(feminine[0].upper() + feminine[1:] if ordinal[0].isupper() and not article else feminine)
        word = "paragens" if noun.lower() == "stops" else "paragem"
        words.append(word[0].upper() + word[1:] if noun[0].isupper() and not words else word)
        return " ".join(words)

    return _PT_STOP_RE.sub(replace, text or "")


def _note_words(text: str) -> set[str]:
    """Return the content words of a note, for near-duplicate detection."""
    return {word for word in normalize_text(text).split() if len(word) >= 4}


def _repeats_a_note(text: str, notes: Sequence[str]) -> bool:
    """Return whether a note says what another note already says.

    Two notes repeat each other when most of the shorter one's words are in
    the other (at least four words and 70%).
    """
    words = _note_words(text)
    if not words:
        return False
    for note in notes:
        other = _note_words(note)
        shared = len(words & other)
        if other and shared >= 4 and shared / min(len(words), len(other)) >= 0.7:
            return True
    return False


_TEMPERATURE_RE = re.compile(r"(\d{1,2}(?:[.,]\d)?)\s*°\s*C")


def _restates_forecast(text: str, strategy: Sequence[str]) -> bool:
    """Return whether a stop's weather note only repeats the plan's forecast.

    A stop note that quotes the same temperatures as the weather section
    ("partly cloudy, 16.8°C to 31.3°C; the viewpoint still works") says again
    what the section already says for the whole plan.
    """
    temperatures = {value.replace(",", ".") for value in _TEMPERATURE_RE.findall(text or "")}
    if not temperatures:
        return False
    stated = {value.replace(",", ".") for line in strategy for value in _TEMPERATURE_RE.findall(line)}
    return temperatures <= stated


def clean_plan_notes(draft: PlanDraft, brief: Optional[PlanBrief], language: str) -> PlanDraft:
    """Tidy the model's free-text notes before the legs are merged.

    Notes about legs or connections are dropped (the legs are computed and
    checked by code, which adds its own note for a leg it cannot confirm),
    and so are notes that talk about the system's internals. Live departures
    and service-hours notices are removed unless the plan starts now. Every
    remaining note is deduplicated and written as a sentence.

    Args:
        draft: Plan with curated blocks.
        brief: Planning brief.
        language: Output language.

    Returns:
        The same draft with cleaned notes.
    """
    live_ok = plan_starts_now(draft, brief)
    if language == "pt":
        draft.direct_answer = _portuguese_stop_word(draft.direct_answer)
        draft.movement_logic = [_portuguese_stop_word(line) for line in draft.movement_logic]
        for block in draft.blocks:
            block.purpose = _portuguese_stop_word(block.purpose)

    def clean(items: List[str], *, drop_connection_notes: bool = True, previous: Sequence[str] = ()) -> List[str]:
        output: List[str] = []
        seen: set[str] = set()
        for item in items:
            if language == "pt":
                item = _portuguese_stop_word(item)
            text = item if live_ok else _strip_live_info(item)
            if not text:
                continue
            normalized = normalize_text(text)
            if drop_connection_notes and _is_connection_note(text):
                continue
            if _INTERNAL_NOTE_RE.search(without_public_cards(normalized)):
                continue
            if normalized in seen or _repeats_a_note(text, [*previous, *output]):
                continue
            seen.add(normalized)
            output.append(_sentence_case(text))
        return output

    # Each note is said once: a weather caveat is not repeated as a final note or a tip.
    draft.weather_strategy = clean(draft.weather_strategy, drop_connection_notes=False)
    draft.limitations = clean(draft.limitations, previous=draft.weather_strategy)
    draft.tips = clean(draft.tips, previous=[*draft.weather_strategy, *draft.limitations])
    draft.direct_answer = _sentence_case(draft.direct_answer if live_ok else _strip_live_info(draft.direct_answer))
    if not live_ok:
        draft.movement_logic = [line for line in (_strip_live_info(line) for line in draft.movement_logic) if line]
    for block in draft.blocks:
        block.weather = [
            note
            for note in clean(block.weather, drop_connection_notes=False, previous=draft.weather_strategy)
            if not _restates_forecast(note, draft.weather_strategy)
        ]
        block.limitations = clean(block.limitations, previous=block.weather)
        purpose = block.purpose if not _INTERNAL_NOTE_RE.search(without_public_cards(normalize_text(block.purpose))) else ""
        block.purpose = _sentence_case(purpose) if purpose else ""
    return draft


def plan_sources(draft: PlanDraft, evidence: EvidenceBundle) -> Dict[str, SourceRef]:
    """Return the sources the renderer may cite and set the plan-level ones.

    Stop sources come from each stop's card. The plan-level sources are the
    ones behind what the plan shows beyond the stops: the operators of the
    resolved legs and IPMA when the plan has a weather section. Sources the
    model listed without using them are dropped.
    """
    sources = dict(evidence.sources)
    movement_text = normalize_text(" ".join(draft.movement_logic))
    used: List[str] = []
    if re.search(r"\bmetro\b", movement_text):
        used.append("metro")
    # Carris Metropolitana runs four-digit lines ("1604"); Carris (Lisbon)
    # runs the trams and the two- and three-digit buses ("728", "15E").
    metropolitana = bool(
        re.search(r"\bcarris metropolitana\b", movement_text)
        or re.search(r"\b(?:bus|autocarro|linha|line)\s+\d{4}\b", movement_text)
    )
    urban_text = re.sub(r"\bcarris metropolitana\b", " ", movement_text)
    urban = bool(
        re.search(r"\b(?:tram|electrico|eletrico)\b", urban_text)
        or re.search(r"\bcarris\b", urban_text)
        or re.search(r"\b(?:bus|autocarro)\s+(?:n\.?\s*)?\d{2,3}[a-z]?\b", urban_text)
        or (re.search(r"\b(?:bus|autocarro)\b", urban_text) and not metropolitana)
    )
    if urban:
        used.append("carris")
    if metropolitana:
        used.append("carris_metropolitana")
    if re.search(r"\b(?:train|comboio|cp)\b", movement_text):
        used.append("cp")
    for source_id in used:
        sources.setdefault(source_id, SOURCE_CATALOG[source_id])
    if draft.weather_strategy and any(card.kind == "weather" for card in evidence.cards):
        used.append("ipma")
        sources.setdefault("ipma", SOURCE_CATALOG["ipma"])
    draft.source_ids = list(dict.fromkeys(used))
    return sources


def other_event_options(draft: PlanDraft, evidence: EvidenceBundle, language: str, limit: int = 4) -> List[str]:
    """Return short lines for the events found but not used in the plan.

    A request such as "what events are on this weekend? pick one and plan the
    evening" asks for the list as well as the plan.
    """
    used = {normalize_text(strip_block_time(block.title)) for block in draft.blocks}
    lines: List[str] = []
    for card in evidence.cards:
        if card.kind != "event" or any(normalize_text(card.title) in key or key in normalize_text(card.title) for key in used if key):
            continue
        fields = getattr(card, "fields", {}) or {}
        when = str(fields.get("When") or "").strip()
        link = str(fields.get("More details") or "").strip()
        # Only real events: a dated card or a VisitLisboa events link.
        if not when and not re.search(r"/(?:events|eventos)/", link):
            continue
        parts = [f"**{card.title}**"]
        if when:
            parts.append(when)
        if link:
            parts.append(link)
        lines.append(" · ".join(parts))
        if len(lines) >= limit:
            break
    return lines


_MEAL_LABELS = {
    "pt": {"lunch": "Almoço", "dinner": "Jantar", "coffee": "Café", "pastry": "Pastelaria"},
    "en": {"lunch": "Lunch", "dinner": "Dinner", "coffee": "Coffee", "pastry": "Pastry"},
}


_FOUND_FOR_ROLE_RES = (
    ("coffee", re.compile(r"\b(?:cafe|cafes|coffee)\b")),
    ("pastry", re.compile(r"\b(?:pastry|pastries|pastelaria|pastelarias|pasteis|confeitaria)\b")),
    ("lunch", re.compile(r"\b(?:lunch|almoco)\b")),
    ("dinner", re.compile(r"\b(?:dinner|jantar)\b")),
    ("meal", re.compile(r"\b(?:restaurants?|restaurantes?)\b")),
)


def label_meal_stops(draft: PlanDraft, evidence: EvidenceBundle, language: str) -> PlanDraft:
    """Prefix food stops with their role ("12:30 · Almoço: Café Paris").

    The role comes from the search that found the stop ("cafés near Chiado"
    makes a coffee stop, whatever the model called it); a generic restaurant
    search is a lunch or a dinner by the time of day. The label makes the plan
    easy to scan and lets follow-up questions such as "where was the lunch?"
    find the stop. Run it after the legs are resolved, since the legs match
    stops by their card title.
    """
    labels = _MEAL_LABELS["pt" if language == "pt" else "en"]
    used: set[tuple[int, str]] = set()
    for block in draft.blocks:
        card = match_block_to_card(block.title, evidence)
        found_for = normalize_text(str(((getattr(card, "fields", {}) or {}) if card is not None else {}).get("Found for", "")))
        role = next((name for name, pattern in _FOUND_FOR_ROLE_RES if pattern.search(found_for)), "")
        if not role and not found_for:
            # Only a card with no search role falls back to the model's kind: a
            # cinema found by the museum search is not a lunch stop.
            kind = normalize_text(block.kind)
            role = {"coffee": "coffee", "cafe": "coffee", "pastry": "pastry"}.get(kind, "meal" if kind in {"food", "restaurant", "lunch", "dinner"} else "")
        minutes = _block_start_minutes(block)
        label = ""
        if role in {"lunch", "dinner"} and minutes is not None:
            # The time decides: a lunch search's restaurant at 19:55 is dinner.
            role = "meal"
        if role in {"coffee", "pastry", "lunch", "dinner"}:
            label = labels[role]
        elif role == "meal" and minutes is not None:
            if 11 * 60 <= minutes < 16 * 60:
                label = labels["lunch"]
            elif minutes >= 18 * 60:
                label = labels["dinner"]
        name = strip_block_time(block.title)
        if not label or normalize_text(label) in normalize_text(name):
            continue
        # One lunch and one dinner a day: a fado house found by the dinner
        # search before the dinner stop is not a second dinner.
        key = (int(getattr(block, "day", 0) or 1), label)
        if label in {labels["lunch"], labels["dinner"]} and key in used:
            continue
        used.add(key)
        block.title = f"{block.title[: len(block.title) - len(name)]}{label}: {name}"
    return draft


# A meal stop closed at its time is replaced by an open restaurant this close to it.
_MEAL_SUBSTITUTE_RADIUS_KM = 1.5


def _open_meal_substitute(
    block: PlanBlock,
    card: Any,
    draft: PlanDraft,
    evidence: EvidenceBundle,
    brief: Optional[PlanBrief],
    language: str,
    start: int,
    block_day: Any,
    today: Any,
) -> Optional[PlanBlock]:
    """Return the same food stop at the nearest place of its type open at that time, or ``None``.

    Dropping a closed lunch leaves a full day with no lunch between the
    morning and the afternoon stops, and dropping a closed café leaves "a
    walk with a church and a café" without the café. Another card of the same
    type (café, pastry shop, or restaurant) open at that time, near the
    closed one and not already in the plan, keeps the stop; other stops are
    dropped as before.
    """
    food_kind = next((kind for kind in ("cafe", "pastry", "restaurant") if card is not None and card_matches_kind(card, kind)), "")
    if not food_kind:
        return None
    origin = card_coordinates(card)
    if not origin:
        return None
    used = {
        normalize_text(getattr(match_block_to_card(other.title, evidence), "title", ""))
        for other in draft.blocks
    }
    candidates = []
    for other in evidence.cards:
        if other is card or normalize_text(other.title) in used or not card_matches_kind(other, food_kind):
            continue
        hours = hours_for_plan_day(str((getattr(other, "fields", {}) or {}).get("Hours", "")), block_day, today)
        point = card_coordinates(other)
        if open_at(hours, start) is not True or not point:
            continue
        distance = haversine_distance(*origin, *point)
        if distance <= _MEAL_SUBSTITUTE_RADIUS_KM:
            candidates.append((distance, other))
    if not candidates:
        return None
    replacement = min(candidates, key=lambda item: item[0])[1]
    old_name = strip_block_time(block.title)
    substitute = copy.copy(block)
    substitute.title = f"{block.title[: len(block.title) - len(old_name)]}{replacement.title}"
    substitute.details = []
    if normalize_text(old_name) in normalize_text(block.purpose or ""):
        substitute.purpose = ""
    curated = curate_block_details(PlanDraft(title="", direct_answer="", blocks=[substitute]), evidence, brief, language)
    if not curated.blocks:
        return None
    # Notes written with the closed restaurant now name the open one.
    for attribute in ("direct_answer",):
        setattr(draft, attribute, str(getattr(draft, attribute) or "").replace(old_name, replacement.title))
    for attribute in ("tips", "limitations", "movement_logic"):
        setattr(draft, attribute, [item.replace(old_name, replacement.title) for item in getattr(draft, attribute)])
    return curated.blocks[0]


# Accent-free words a note uses to refer to a stop by its type.
_STOP_TYPE_NOTE_WORDS = {
    "cafe": ("cafe", "cafes", "coffee", "esplanada"),
    "pastry": ("pastelaria", "pastelarias", "pastry"),
    "museum": ("museu", "museus", "museum", "museums"),
    "viewpoint": ("miradouro", "miradouros", "viewpoint", "viewpoints"),
    "garden": ("jardim", "jardins", "garden", "gardens"),
    "beach": ("praia", "praias", "beach", "beaches"),
    "market": ("mercado", "mercados", "market", "markets"),
}


def drop_closed_stops(draft: PlanDraft, evidence: EvidenceBundle, brief: Optional[PlanBrief], language: str) -> PlanDraft:
    """Remove stops scheduled when their card shows them closed, with a note.

    The review asks the model to move such stops; this is the backstop when it
    does not. Card hours are today's, so only plans for today are checked.
    """
    today = lisbon_now().date()
    day_of_plan = plan_date(brief, today)
    kept: List[PlanBlock] = []
    dropped: List[str] = []
    dropped_cards: List[Any] = []
    substituted = False
    for block in draft.blocks:
        card = match_block_to_card(block.title, evidence)
        start = _block_start_minutes(block)
        hours = str((getattr(card, "fields", {}) or {}).get("Hours", "")) if card is not None else ""
        block_day = day_of_plan + timedelta(days=max(0, int(getattr(block, "day", 0) or 1) - 1))
        hours = hours_for_plan_day(hours, block_day, today)
        if start is not None and open_at(hours, start) is False:
            substitute = _open_meal_substitute(block, card, draft, evidence, brief, language, start, block_day, today)
            if substitute is not None:
                kept.append(substitute)
                substituted = True
                continue
            dropped.append(strip_block_time(block.title))
            dropped_cards.append(card)
            continue
        kept.append(block)
    if not kept:
        return draft
    if not dropped:
        # A swapped meal stop (its notes and direct answer already updated)
        # must still replace the closed one in the stop list.
        if substituted:
            draft.blocks = kept
        return draft
    draft.blocks = kept
    # Notes and the direct answer were written with the dropped stop in them.
    dropped_tokens = {
        token for name in dropped for token in normalize_text(name).split() if len(token) >= 5
    }

    # A note may name the dropped stop only by its type ("o café escolhido
    # fica fora de Alfama"); when no kept stop has that type, it is stale too.
    kept_cards_now = [match_block_to_card(block.title, evidence) for block in kept]
    lost_type_words = {
        word
        for dropped_card in dropped_cards
        for kind, words in _STOP_TYPE_NOTE_WORDS.items()
        if dropped_card is not None
        and card_matches_kind(dropped_card, kind)
        and not any(other is not None and card_matches_kind(other, kind) for other in kept_cards_now)
        for word in words
    }

    def mentions_dropped(text: str) -> bool:
        words = set(normalize_text(text).split())
        return bool(dropped_tokens & words or lost_type_words & words)

    draft.tips = [tip for tip in draft.tips if not mentions_dropped(tip)]
    draft.limitations = [item for item in draft.limitations if not mentions_dropped(item)]
    if mentions_dropped(draft.direct_answer):
        # Rewritten like the planner's own direct answers: where, from when, which stops.
        names = [strip_block_time(block.title) for block in kept]
        listed = ", ".join(names[:-1]) + (" e " if language == "pt" else " and ") + names[-1] if len(names) > 1 else names[0]
        first_start = _block_start_minutes(kept[0])
        when = f"{first_start // 60:02d}:{first_start % 60:02d}" if first_start is not None else ""
        area = brief.activity_areas[0] if brief and brief.activity_areas else ""
        if language == "pt":
            lead = ", ".join(part for part in (f"Em {area}" if area else "", f"a partir das {when}" if when else "") if part)
            draft.direct_answer = f"{lead}, passas por {listed}." if lead else f"O plano passa por {listed}."
        else:
            lead = ", ".join(part for part in (f"In {area}" if area else "", f"from {when}" if when else "") if part)
            draft.direct_answer = f"{lead}, you visit {listed}." if lead else f"The plan takes you to {listed}."
    # A closed stop the user asked for, or the only stop of a requested
    # type, is worth a note; any other stop is simply left out.
    named = [normalize_text(place) for place in (brief.named_places if brief else []) if place]
    requested_kinds = [component.kind for component in (brief.components if brief else [])]
    kept_cards = [match_block_to_card(block.title, evidence) for block in kept]
    for name, card in zip(dropped, dropped_cards):
        title = normalize_text(name)
        user_named = any(place in title or title in place for place in named)
        lost_kind = card is not None and any(
            kind not in {"attraction", "walk", "event"}
            and card_matches_kind(card, kind)
            and not any(other is not None and card_matches_kind(other, kind) for other in kept_cards)
            for kind in requested_kinds
        )
        if not (user_named or lost_kind):
            continue
        draft.limitations.append(
            f"{name} está fechado a essa hora no dia do plano, por isso ficou fora do plano."
            if language == "pt"
            else f"{name} is closed at that time on the day of the plan, so it was left out."
        )
    return draft


def drop_closed_cards(evidence: EvidenceBundle, brief: PlanBrief) -> EvidenceBundle:
    """Remove the cards of places closed all day on a one-day plan's day.

    A stop the planner picks and the hours check then drops leaves the plan
    short ("a walk with a bookshop and a café" reduced to the café). A place
    the user named is kept, so the plan can say it is closed.
    """
    if brief.days > 1:
        return evidence
    today = lisbon_now().date()
    day = plan_date(brief, today)
    named = [normalize_text(place) for place in brief.named_places if place]
    kept = []
    for card in evidence.cards:
        title = normalize_text(card.title)
        if (
            card.kind in {"place", "food"}
            and closed_all_day(hours_for_plan_day(str((card.fields or {}).get("Hours", "")), day, today))
            and not any(place in title or title in place for place in named)
        ):
            continue
        kept.append(card)
    evidence.cards = kept
    return evidence


# Catalogue categories of businesses that sell an activity (tour operators,
# cruise and diving companies, golf clubs, event venues) rather than a place
# to visit, in English and in the Portuguese labels of the cards.
_OPERATOR_CATEGORY_RE = re.compile(
    r"\b(?:tours|sightseeing|adventure tours|water sports|desportos aquaticos|tejo cruises|cruzeiros no tejo|"
    r"wine tourism|golf|golfe|holes|buracos|cultural centres trips|trips|venues|marinas|ports|dmc|pco|meeting facilities)\b"
)
# Operator names that other categories hide ("Hippotrip - Turismo Anfíbio" is "Family & Kids").
_OPERATOR_TITLE_RE = re.compile(r"\b(?:tours?|turismo|experiences|excursoes|cruises|cruzeiros|charter|travel|trips)\b")
# Requests that ask for such an activity keep those cards.
_OPERATOR_REQUEST_RE = re.compile(
    r"\b(?:tour|tours|cruise|cruzeiro|boat|barco|dive|diving|mergulho|surf|golf|golfe|kayak|caiaque|"
    r"wine|vinho|tasting|prova|sail|sailing|vela|guided|guiada|excursion|excursao|passeio de barco)\b"
)
# Accommodation (hotels, hostels, pousadas, campsites) and tourist offices are
# where a visitor sleeps or asks for information, not stops of an itinerary.
_LODGING_CATEGORY_RE = re.compile(
    r"\b(?:hot[ae]is|hotels?|aparthoteis|apartments|apartamentos|accommodation|alojamento|guest houses?|"
    r"hostels?|pousadas?|camping|campismo|tourist offices?|postos? de turismo)\b"
)
_LODGING_TITLE_RE = re.compile(r"\b(?:hotel|hostel|pousada|aparthotel|guest ?house|camping|parque de campismo)\b")
_LODGING_REQUEST_RE = re.compile(
    r"\b(?:hotel|hoteis|hostel|pousada|alojamento|accommodation|lodging|stay|ficar|dormir|sleep|"
    r"check[- ]?in|tourist office|posto de turismo|informacao turistica)\b"
)
# (category pattern, title pattern, request pattern that keeps the family).
_NON_STOP_FAMILIES = (
    (_OPERATOR_CATEGORY_RE, _OPERATOR_TITLE_RE, _OPERATOR_REQUEST_RE),
    (_LODGING_CATEGORY_RE, _LODGING_TITLE_RE, _LODGING_REQUEST_RE),
)


def drop_operator_cards(evidence: EvidenceBundle, brief: PlanBrief) -> EvidenceBundle:
    """Remove cards that are not itinerary stops unless the request asks for them.

    Two families are covered: activity operators ("somewhere off the beaten
    track" found two tour-company offices; an office is not a stop) and
    accommodation or tourist offices (a pousada suggested as a stop of a
    walk). A request for a boat tour keeps the operators; a request that
    mentions the hotel or the stay keeps the lodging cards. A place the user
    named is always kept.

    Args:
        evidence: Evidence bundle whose cards are filtered in place.
        brief: Structured brief of the request.

    Returns:
        The same bundle, without the cards of families the request did not ask for.
    """
    request = normalize_text(
        " ".join(
            [*(f"{component.kind} {component.detail}" for component in brief.components), *brief.preferences]
        )
    )
    families = [
        (category_re, title_re)
        for category_re, title_re, request_re in _NON_STOP_FAMILIES
        if not request_re.search(request)
    ]
    if not families:
        return evidence
    named = [normalize_text(place) for place in brief.named_places if place]
    kept = []
    for card in evidence.cards:
        title = normalize_text(card.title)
        category = normalize_text(str((card.fields or {}).get("Category", "")))
        if (
            card.kind in {"place", "food"}
            and any(
                (category and category_re.search(category)) or title_re.search(title)
                for category_re, title_re in families
            )
            and not any(place in title or title in place for place in named)
        ):
            continue
        kept.append(card)
    evidence.cards = kept
    return evidence


def drop_topic_cards(evidence: EvidenceBundle, brief: PlanBrief) -> EvidenceBundle:
    """Remove place cards with no address before the planner sees them.

    Topic pages ("Fado", "Lisbon's Azulejos") describe a theme, not a place to
    go, and read as a stop with no address. A place the user named is kept.
    """
    named = [normalize_text(place) for place in brief.named_places if place]
    kept = []
    for card in evidence.cards:
        fields = card.fields or {}
        title = normalize_text(card.title)
        if any(place in title or title in place for place in named):
            kept.append(card)
            continue
        if card.kind in {"place", "food"} and not any(fields.get(key) for key in ("Address", "Venue", "Coordinates")):
            continue
        # The address may carry a map link whose coordinates are digits; only
        # its visible text says whether it is a street or an area.
        visible_address = re.sub(r"\]\([^)]*\)|https?://\S+", " ", str(fields.get("Address") or ""))
        if card.kind == "place" and _is_area_or_topic_card(title, normalize_text(visible_address)):
            continue
        kept.append(card)
    evidence.cards = kept
    return evidence


# Words that make a card title a place to go (a venue), not a district or a theme.
_VENUE_TITLE_RE = re.compile(
    r"\b(?:museu|museum|mosteiro|monastery|torre|tower|palacio|palace|miradouro|viewpoint|jardim|garden|"
    r"igreja|church|se|cathedral|castelo|castle|mercado|market|parque|park|centro|center|centre|teatro|theatre|"
    r"restaurante|restaurant|cafe|pastelaria|bar|livraria|bookshop|oceanario|aquario|aqueduto|convento|basilica|"
    r"padrao|monumento|monument|elevador|funicular|estacao|station|praia|beach|fundacao|galeria|gallery|casa|house|"
    r"panteao|capela|chapel|arco|fabrica|factory|pavilhao|pavilion|estadio|stadium|zoo|jardim zoologico)\b"
)
# Street words that make an address a street address rather than an area name.
_STREET_ADDRESS_RE = re.compile(
    r"\d|\b(?:rua|r\.|avenida|av\.?|largo|praca|travessa|calcada|estrada|campo|cais|terreiro|alameda|rotunda|"
    r"beco|escadinhas|parque|jardim|quinta|bairro\s+\w+\s+\d)\b"
)


def _is_area_or_topic_card(title: str, address: str) -> bool:
    """Return whether a place card names a district or a theme rather than a venue.

    VisitLisboa lists pages such as "Belém" (a district) and "Calçada
    Portuguesa" (a paving tradition) as places; their address is only an area
    ("Baixa"), and they read as stops with nothing to visit at a set point.

    Args:
        title: Normalized card title.
        address: Normalized card address.

    Returns:
        True when the title names no venue and the address is not a street address.
    """
    if not title or _VENUE_TITLE_RE.search(title):
        return False
    return not address or not _STREET_ADDRESS_RE.search(address)


def drop_duplicate_stops(draft: PlanDraft) -> PlanDraft:
    """Remove repeated stops (same card chosen twice)."""
    seen: set[str] = set()
    unique = []
    for block in draft.blocks:
        key = normalize_text(strip_block_time(block.title))
        if key in seen:
            continue
        seen.add(key)
        unique.append(block)
    draft.blocks = unique
    return draft


def repair_messages(messages: Sequence[Any], raw_answer: str, issues: Sequence[str]) -> list:
    """Return the messages for one repair round of the planner LLM."""
    from langchain_core.messages import AIMessage, HumanMessage

    return [
        *messages,
        AIMessage(content=raw_answer),
        HumanMessage(
            content=(
                "Revise the JSON plan to fix these issues, keeping everything else that is correct. "
                "Do not mention the corrections, the changed times, or the reasons for them in any text field; "
                "write the plan as if it had been right from the start. "
                "Return the complete corrected JSON only.\n- " + "\n- ".join(issues[:10])
            )
        ),
    ]
