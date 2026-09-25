# ==========================================================================
# Master Thesis - Itinerary Leg Formatting
#   - André Filipe Gomes Silvestre, 20240502
#
#   Pure helpers that turn transport-tool evidence into one movement line per
#   itinerary leg: walking legs from straight-line distances, and Metro legs
#   from the route text returned by the Metro routing tool. The Transport
#   worker calls the tools; these helpers only read what the tools returned,
#   so a leg line never contains a line, station, or duration that the
#   evidence does not show.
# ==========================================================================

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple


# Straight-line distance at or below which two stops are treated as walkable.
WALKABLE_DISTANCE_KM = 1.2
# Longest walk offered when no public-transport line connects two stops.
MAX_FALLBACK_WALK_KM = 2.5
# Urban walking pace on the straight-line distance, with a detour factor:
# about 5 km/h over a path 25% longer than the straight line.
_WALK_MINUTES_PER_STRAIGHT_KM = 15.0
_METRO_LINE_NAMES_PT = {"blue": "Azul", "green": "Verde", "red": "Vermelha", "yellow": "Amarela"}
_METRO_LINE_NAMES_EN = {"azul": "Blue", "verde": "Green", "vermelha": "Red", "amarela": "Yellow"}
_METRO_LINE_EMOJIS = {"🔵": "Blue", "🟢": "Green", "🔴": "Red", "🟡": "Yellow"}


@dataclass
class LegRequest:
    """One movement between two consecutive points of a plan.

    Attributes:
        origin_label: User-facing name of the start of the leg.
        origin_query: Address or name used to locate the start.
        destination_label: User-facing name of the end of the leg.
        destination_query: Address or name used to locate the end.
        prefer_transit: Whether the user's requested mode should be tried
            before walking (for the leg from the start point).
        origin_coords: Stored coordinates of the start, when known.
        destination_coords: Stored coordinates of the end, when known.
        walk_only: Whether to return the walking distance and time whatever
            the distance (a last mile from a train station).
    """

    origin_label: str
    origin_query: str
    destination_label: str
    destination_query: str
    prefer_transit: bool = False
    origin_coords: Optional[Tuple[float, float]] = None
    destination_coords: Optional[Tuple[float, float]] = None
    walk_only: bool = False


_LEG_MINUTES_RE = re.compile(r"(\d{1,3})\s*min\b")
# Marker the Metro routing tool writes when the route rides a stopped section.
_METRO_INTERRUPTED_RE = re.compile(r"⛔ \*\*Interrupted on this route \((\w+) Line\)\*\*")


def metro_route_interrupted_lines(route_text: str, language: str = "pt") -> List[str]:
    """Return the localized names of the Metro lines a route rides while they are stopped.

    Args:
        route_text: Output of the Metro routing tool.
        language: Output language (``pt`` or ``en``).

    Returns:
        Line names such as ``["Linha Amarela"]``; empty when the route is clear.
    """
    names: List[str] = []
    for line in _METRO_INTERRUPTED_RE.findall(str(route_text or "")):
        localized = _localize_line(line, language)
        name = f"Linha {localized}" if language == "pt" else f"{localized} Line"
        if name not in names:
            names.append(name)
    return names


_ALTERNATIVE_RE = re.compile(r"\((?:alternativa|alternative)s?:[^)]*\)", re.IGNORECASE)


def localize_departure_notes(text: str, language: str) -> str:
    """Translate the Carris live-departure notes, which the tool writes in Portuguese.

    Args:
        text: Departure text such as "11:12 (tempo real), 11:22 (11 min de atraso)".
        language: Output language; Portuguese text is returned unchanged.

    Returns:
        The text with "live", "N min late", and "N min early" notes in English.
    """
    if language == "pt":
        return text
    text = re.sub(r"\(tempo real\)", "(live)", str(text or ""), flags=re.IGNORECASE)
    text = re.sub(r"\((\d+) min de atraso\)", r"(\1 min late)", text, flags=re.IGNORECASE)
    return re.sub(r"\((\d+) min adiantad[oa]\)", r"(\1 min early)", text, flags=re.IGNORECASE)


def estimate_leg_minutes(line: str) -> Optional[int]:
    """Return the door-to-door minutes a leg line states (walks plus ride).

    The leg formatters below state every part of a leg as "N min" (walk to
    the stop, ride, walk to the destination); alternatives in parentheses
    are ignored.

    Args:
        line: One rendered leg line.

    Returns:
        Total minutes, or ``None`` when the line states no duration.
    """
    minutes = [int(value) for value in _LEG_MINUTES_RE.findall(_ALTERNATIVE_RE.sub("", str(line or "")))]
    return sum(minutes) if minutes else None


def walking_minutes(distance_km: float) -> int:
    """Return an estimated walking time, rounded to whole minutes (minimum 3)."""
    return max(3, int(round(distance_km * _WALK_MINUTES_PER_STRAIGHT_KM)))


def format_walking_leg(origin_label: str, destination_label: str, distance_km: float, language: str) -> str:
    """Render a walking leg with the measured distance and estimated time."""
    minutes = walking_minutes(distance_km)
    meters = int(round(distance_km * 1000 / 50.0) * 50) or 50
    distance_text = f"{meters} m" if meters < 1000 else f"{distance_km:.1f} km".replace(".", "," if language == "pt" else ".")
    if language == "pt":
        return f"🚶 **{origin_label} → {destination_label}:** a pé, cerca de {minutes} min ({distance_text} em linha reta)."
    return f"🚶 **{origin_label} → {destination_label}:** walk, about {minutes} min ({distance_text} in a straight line)."


def _station_name(value: str) -> str:
    """Restore lower-case Portuguese particles in title-cased station names."""
    return re.sub(r"\b(De|Do|Da|Dos|Das|E)\b", lambda match: match.group(1).lower(), value.strip())


def _localize_line(line: str, language: str) -> str:
    """Return the Metro line name in the output language."""
    key = line.strip().lower()
    if language == "pt":
        return _METRO_LINE_NAMES_PT.get(key, line.strip().title())
    return _METRO_LINE_NAMES_EN.get(key, line.strip().title())


def summarize_metro_route(
    route_text: str,
    origin_label: str,
    destination_label: str,
    language: str,
    *,
    walk_before_km: Optional[float] = None,
    walk_after_km: Optional[float] = None,
) -> str:
    """Summarize a Metro route answer into one leg line.

    The line keeps what a traveller needs: the boarding station and
    direction, the transfer station and second line when there is one, the
    exit station, the estimated time, and the walks to and from the stations
    when they are longer than a couple of minutes. Live waiting times are
    left out because they are only valid for an immediate departure.

    Args:
        route_text: Output of the Metro routing tool.
        origin_label: User-facing start label.
        destination_label: User-facing end label.
        language: Output language (``pt`` or ``en``).
        walk_before_km: Straight-line distance from the start to the boarding station.
        walk_after_km: Straight-line distance from the exit station to the end.

    Returns:
        One Markdown leg line, or an empty string when the text holds no
        usable Metro route.
    """
    text = str(route_text or "")
    if "METRO ROUTE" not in text.upper():
        return ""
    boards = re.findall(r"Board at \*\*([^*]+)\*\*(?:\s*(?:→\s*)?(?:direção|direction|towards)?\s*\*\*([^*]+)\*\*)?", text)
    exits = re.findall(r"Exit at \*\*([^*]+)\*\*", text)
    if not boards or not exits:
        return ""
    time_match = re.search(r"Estimated travel time:\s*\*\*([^*]+)\*\*", text)
    minutes = time_match.group(1).strip() if time_match else ""
    first_line_match = re.search(r"Take \*\*(\w+) Line\*\*", text)
    transfer_match = re.search(r"Transfer at\*\*:\s*([^(\n]+)", text)
    second_line_match = re.search(r"Transfer to \*\*(\w+) Line[^*]*\*\*\s*(?:→\s*)?(?:direção|direction|towards)?\s*\*\*([^*]+)\*\*", text)
    board_station, board_direction = _station_name(boards[0][0]), _station_name(boards[0][1])
    exit_station = _station_name(exits[-1])
    is_pt = language == "pt"

    if transfer_match and second_line_match:
        transfer_station = _station_name(transfer_match.group(1))
        second_line = _localize_line(second_line_match.group(1), language)
        second_direction = _station_name(second_line_match.group(2))
        # "Transfer at: Alameda (🟢 ↔ 🔴)": the first emoji is the boarding line.
        first_emoji = re.search(r"Transfer at\*\*:\s*[^(\n]+\((\S+?)\ufe0f?\s*↔", text)
        first_line = _METRO_LINE_EMOJIS.get(first_emoji.group(1), "") if first_emoji else ""
        if is_pt:
            first = f"Metro, Linha {_localize_line(first_line, language)}, de" if first_line else "Metro desde"
            body = (
                f"{first} **{board_station}** (direção {board_direction}), transbordo em **{transfer_station}** "
                f"para a Linha {second_line} (direção {second_direction}) e saída em **{exit_station}**"
            )
        else:
            first = f"Metro, {first_line} Line, from" if first_line else "Metro from"
            body = (
                f"{first} **{board_station}** (towards {board_direction}), change at **{transfer_station}** "
                f"to the {second_line} Line (towards {second_direction}) and exit at **{exit_station}**"
            )
    else:
        line = _localize_line(first_line_match.group(1), language) if first_line_match else ""
        if is_pt:
            direction = f" (direção {board_direction})" if board_direction else ""
            body = (
                f"Metro, Linha {line}, de **{board_station}**{direction} até **{exit_station}**"
                if line
                else f"Metro de **{board_station}**{direction} até **{exit_station}**"
            )
        else:
            line_text = f"{line} Line" if line else "Metro"
            direction = f" (towards {board_direction})" if board_direction else ""
            body = f"Metro, {line_text}, from **{board_station}**{direction} to **{exit_station}**" if line else f"Metro from **{board_station}**{direction} to **{exit_station}**"
    if minutes:
        body += f", {minutes}"
    interrupted = metro_route_interrupted_lines(text, language)
    if interrupted:
        names = " e ".join(interrupted) if is_pt else " and ".join(interrupted)
        body += (
            f" (atenção: {names} com circulação interrompida neste momento)"
            if is_pt
            else f" (note: {names} currently interrupted)"
        )
    if walk_before_km is not None and walk_before_km > 0.15:
        before = walking_minutes(walk_before_km)
        body = (
            f"a pé até **{board_station}** (cerca de {before} min), depois " + body[0].lower() + body[1:]
            if is_pt
            else f"walk to **{board_station}** (about {before} min), then " + body[0].lower() + body[1:]
        )
    if walk_after_km is not None and walk_after_km > 0.15:
        after = walking_minutes(walk_after_km)
        body += (
            f", e depois a pé até {destination_label} (cerca de {after} min)"
            if is_pt
            else f", then walk to {destination_label} (about {after} min)"
        )
    return f"🚇 **{origin_label} → {destination_label}:** {body}."


# Wait assumed when a line lists no departure times (about half a typical headway).
_UNKNOWN_WAIT_MINUTES = 10
_DEPARTURES_LINE_RE = re.compile(
    r"(?:Pr[oó]ximas partidas|Next departures|Next)(?:\*\*)?\s*:\s*(?:\*\*)?\s*(?P<body>[^\n]+)",
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(r"\b(?P<hour>\d{1,2}):(?P<minute>\d{2})\b")


def departure_times(text: str) -> List[str]:
    """Return the departure entries of a "next departures" line, with their live notes.

    Args:
        text: A line such as "Próximas partidas: 11:12 (tempo real), 11:21, 11:33 (paragem X)".

    Returns:
        Entries such as ``["11:12 (tempo real)", "11:21", "11:33"]``; empty when none.
    """
    match = _DEPARTURES_LINE_RE.search(str(text or ""))
    if not match:
        return []
    body = re.sub(r"\s*\((?:stop|paragem)\s+[^)]*\)\s*$", "", match.group("body"), flags=re.IGNORECASE)
    return [entry.strip() for entry in body.split(",") if _CLOCK_RE.search(entry)]


def wait_minutes_for(departures: List[str], walk_before: Optional[int], now: Optional[datetime] = None) -> Optional[int]:
    """Return the wait at the stop for the first departure reachable after the walk there.

    Args:
        departures: Entries from :func:`departure_times`.
        walk_before: Minutes of walking to the boarding stop.
        now: Current time (defaults to now).

    Returns:
        Minutes of waiting, or ``None`` when no listed departure can be reached.
    """
    now = now or datetime.now()
    ready = now.hour * 60 + now.minute + (walk_before or 0)
    for entry in departures:
        clock = _CLOCK_RE.search(entry)
        if not clock:
            continue
        departure = int(clock.group("hour")) * 60 + int(clock.group("minute"))
        delta = departure - ready
        if delta < -12 * 60:
            delta += 24 * 60
        if delta >= 0:
            return delta
    return None


def door_to_door_minutes(
    walk_before: Optional[int],
    ride: Optional[int],
    walk_after: Optional[int],
    departures: List[str],
    now: Optional[datetime] = None,
) -> int:
    """Return walk + wait + ride + walk, the time a traveller actually spends.

    A line with no departure times gets a typical wait; an unknown ride
    time ranks the option last.
    """
    wait = wait_minutes_for(departures, walk_before, now)
    return (
        (walk_before or 0)
        + (wait if wait is not None else _UNKNOWN_WAIT_MINUTES)
        + (ride if ride is not None else 99)
        + (walk_after or 0)
    )


@dataclass
class CarrisOption:
    """One direct Carris route between two points, as reported by the route tool."""

    line: str
    headsign: str = ""
    board: str = ""
    alight: str = ""
    minutes: Optional[int] = None
    walk_before: Optional[int] = None
    walk_after: Optional[int] = None
    departures: List[str] = field(default_factory=list)

    @property
    def is_tram(self) -> bool:
        """Return whether the line is a tram (Carris tram lines end in "E")."""
        return self.line.upper().endswith("E")

    @property
    def door_to_door_minutes(self) -> int:
        """Walk to the stop, wait, ride, and walk to the destination, in minutes."""
        return door_to_door_minutes(self.walk_before, self.minutes, self.walk_after, self.departures)

    def next_departure(self) -> str:
        """Return the first listed departure the traveller can still reach, or ""."""
        wait = wait_minutes_for(self.departures, self.walk_before)
        if wait is None:
            return ""
        now = datetime.now()
        ready = now.hour * 60 + now.minute + (self.walk_before or 0)
        for entry in self.departures:
            clock = _CLOCK_RE.search(entry)
            if clock and (int(clock.group("hour")) * 60 + int(clock.group("minute")) - ready) % (24 * 60) == wait:
                return entry
        return ""


def extract_carris_options(route_text: str) -> List[CarrisOption]:
    """Parse the direct routes listed by the Carris route tool.

    Args:
        route_text: Output of ``carris_find_routes_between``.

    Returns:
        Options with both stops known, fastest door to door first: walk to
        the stop, wait for the next reachable departure, ride, and walk on
        (a closer tram leaving sooner beats a shorter ride from a far stop).
    """
    options: List[CarrisOption] = []
    current: Optional[CarrisOption] = None
    for raw_line in str(route_text or "").splitlines():
        stripped = raw_line.strip()
        route_match = re.match(r"^-\s+\*\*(?P<line>\d{1,4}[A-Z]?)\*\*\s*:\s*(?:para|to|towards)?\s*(?P<headsign>.*)$", stripped, re.IGNORECASE)
        if route_match:
            current = CarrisOption(line=route_match.group("line").upper(), headsign=route_match.group("headsign").strip())
            options.append(current)
            continue
        if current is None:
            continue
        stops_match = re.search(
            r"(?:embarcar em|apanha em|apanhar em|board at)\s+\*\*(?P<board>[^*]+)\*\*.*?"
            r"(?:sair em|sai em|exit at|alight at)\s+\*\*(?P<alight>[^*]+)\*\*",
            stripped,
            re.IGNORECASE,
        )
        if stops_match:
            current.board = stops_match.group("board").strip()
            current.alight = stops_match.group("alight").strip()
        minutes_match = re.search(
            r"(?:Tempo em ve[ií]culo|Tempo estimado|In-vehicle time|Travel time)(?:\*\*)?\s*:\s*(?:\*\*)?\s*~?\s*(\d+)\s*min",
            stripped,
            re.IGNORECASE,
        )
        if minutes_match:
            current.minutes = int(minutes_match.group(1))
        walk_match = re.search(
            r"(?P<which>Caminhada inicial|Initial walk|Caminhada final|Final walk)(?:\*\*)?\s*:\s*(?:\*\*)?\s*~?\s*(?P<minutes>\d+)\s*min",
            stripped,
            re.IGNORECASE,
        )
        if walk_match:
            which = walk_match.group("which").lower()
            if "inicial" in which or "initial" in which:
                current.walk_before = int(walk_match.group("minutes"))
            else:
                current.walk_after = int(walk_match.group("minutes"))
        if not current.departures:
            current.departures = departure_times(stripped)
    usable = [option for option in options if option.board and option.alight]
    return sorted(usable, key=lambda option: option.door_to_door_minutes)


def format_carris_leg(
    options: List[CarrisOption],
    origin_label: str,
    destination_label: str,
    language: str,
    requested_mode: str = "",
) -> str:
    """Render a Carris tram/bus leg from the fastest direct option and one alternative.

    When the user asked for a tram (or a bus) and only the other vehicle
    connects the two points, the leg says so instead of passing it off as
    the requested mode.
    """
    if not options:
        return ""
    is_pt = language == "pt"
    best = options[0]
    mode_note = ""
    if requested_mode == "tram" and not best.is_tram:
        mode_note = "sem elétrico direto confirmado; " if is_pt else "no direct tram confirmed; "
    elif requested_mode == "bus" and best.is_tram:
        mode_note = "sem autocarro direto confirmado; " if is_pt else "no direct bus confirmed; "
    kind = ("elétrico" if best.is_tram else "autocarro") if is_pt else ("tram" if best.is_tram else "bus")
    icon = "🚋" if best.is_tram else "🚌"
    duration = f", ~{best.minutes} min" if best.minutes is not None else ""
    headsign = (f" (direção {best.headsign})" if is_pt else f" (towards {best.headsign})") if best.headsign else ""
    if is_pt:
        body = f"{kind} **{best.line}**{headsign}, de **{best.board}** até **{best.alight}**{duration}"
    else:
        body = f"{kind} **{best.line}**{headsign} from **{best.board}** to **{best.alight}**{duration}"
    alternatives = [option.line for option in options[1:3] if option.line != best.line]
    if alternatives:
        body += (" (alternativa: " if is_pt else " (alternative: ") + ", ".join(alternatives) + ")"
    # Walks of a couple of minutes to or from the stop are not worth a clause.
    if best.walk_before and best.walk_before >= 3:
        body = (
            f"a pé até **{best.board}** (cerca de {best.walk_before} min), depois " + body
            if is_pt
            else f"walk to **{best.board}** (about {best.walk_before} min), then " + body
        )
    if best.walk_after and best.walk_after >= 3:
        body += (
            f", e depois a pé (cerca de {best.walk_after} min)"
            if is_pt
            else f", then walk (about {best.walk_after} min)"
        )
    return f"{icon} **{origin_label} → {destination_label}:** {mode_note}{body}."


def carris_option_sentence(options: List[CarrisOption], language: str, user_message: str = "") -> str:
    """Return a direct-answer sentence for the fastest Carris option, or "".

    A requested vehicle ("de elétrico", "by bus") is preferred; when only the
    other one is listed, the sentence says the requested one was not found.

    Args:
        options: Carris options, fastest first (``extract_carris_options``).
        language: Output language (``pt`` or ``en``).
        user_message: The request, to read a requested vehicle from.

    Returns:
        "Apanha o elétrico **28E** ... cerca de 11 min." or "" without options.
    """
    if not options:
        return ""
    requested = unicodedata.normalize("NFKD", user_message or "").encode("ascii", "ignore").decode("ascii").lower()
    wants_tram = bool(re.search(r"\b(?:eletricos?|electricos?|trams?)\b", requested))
    wants_bus = bool(re.search(r"\b(?:autocarros?|bus(?:es)?)\b", requested))
    is_pt = language == "pt"
    fallback_note = ""
    if wants_tram != wants_bus:
        matching = [option for option in options if option.is_tram == wants_tram]
        if matching:
            options = matching
        elif is_pt:
            fallback_note = "Não encontrei um elétrico direto; em alternativa, " if wants_tram else "Não encontrei um autocarro direto; em alternativa, "
        else:
            fallback_note = "No direct tram was found; instead, " if wants_tram else "No direct bus was found; instead, "
    best = options[0]
    kind = ("o elétrico" if best.is_tram else "o autocarro") if is_pt else ("tram" if best.is_tram else "bus")
    departure = best.next_departure()
    arrival = ""
    clock = _CLOCK_RE.search(departure)
    if clock and best.minutes is not None:
        arrival_minutes = int(clock.group("hour")) * 60 + int(clock.group("minute")) + best.minutes + (best.walk_after or 0)
        arrival = f"{(arrival_minutes // 60) % 24:02d}:{arrival_minutes % 60:02d}"
    if is_pt:
        sentence = f"apanha {kind} **{best.line}**"
        if best.headsign:
            sentence += f" (direção {best.headsign})"
        sentence += f" em **{best.board}** e sai em **{best.alight}**"
        if best.minutes is not None:
            sentence += f", cerca de {best.minutes} min"
        if departure:
            sentence += f"; próxima partida às **{departure}**"
            if arrival:
                sentence += f", chegada prevista por volta das **{arrival}**"
        sentence = fallback_note + sentence if fallback_note else sentence[0].upper() + sentence[1:]
        return sentence + "."
    sentence = f"{'take' if fallback_note else 'Take'} {kind} **{best.line}**"
    if best.headsign:
        sentence += f" (towards {best.headsign})"
    sentence += f" from **{best.board}** to **{best.alight}**"
    if best.minutes is not None:
        sentence += f", about {best.minutes} min"
    if departure:
        sentence += f"; next departure at **{departure}**"
        if arrival:
            sentence += f", arriving around **{arrival}**"
    return fallback_note + sentence + "."


def parse_leg_endpoint(value: Optional[str]) -> str:
    """Strip Markdown link syntax so an address can be geocoded."""
    text = str(value or "").strip()
    link = re.match(r"^\[([^\]]+)\]\([^)]+\)$", text)
    if link:
        text = link.group(1)
    return re.sub(r"\s+", " ", text).strip(" .")


def address_variants(value: Optional[str]) -> List[str]:
    """Return an address and simpler forms of it, in the order to geocode them.

    "Tv. Salitre /Avenida Liberdade, 1269-066, Lisboa" is not found as
    written; "Tv. Salitre, 1269-066, Lisboa" is. The last form keeps only the
    street and the town.
    """
    text = parse_leg_endpoint(value)
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return []
    # "Tv. Salitre /Avenida Liberdade", "Av. Brasília - Edifício Espelho d'Água":
    # the first alternative is the street.
    street = re.split(r"\s+/\s*|\s*/\s+|\s+-\s+", parts[0])[0].strip()
    variants = [text, ", ".join([street, *parts[1:]])]
    if len(parts) > 1:
        town = parts[-1] if not re.search(r"\d", parts[-1]) else "Lisboa"
        variants.append(f"{street}, {town}")
    return list(dict.fromkeys(variant for variant in variants if variant))
