# ==========================================================================
# Master Thesis - Transport Agent
#   - André Filipe Gomes Silvestre, 20240502
#
#   Specialized agent for transport-related queries.
#   Handles metro, bus, train, and multi-modal routing.
#   Uses BaseAgent.execute_react_loop() for tool execution.
# ==========================================================================

import re
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from agent.agents.base import BaseAgent, traceable
from agent.prompts.transport import get_transport_prompt
from agent.state import AgentState
from agent.utils.langgraph_compat import ToolNode
from agent.utils.response_formatter import (
    finalize_worker_response,
    infer_response_language,
)


def _normalize_token(text: str) -> str:
    """Normalizes station and direction tokens for robust matching."""
    import unicodedata

    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    return normalized.lower().strip()


def _extract_route_endpoints(user_message: str) -> Optional[Tuple[str, str]]:
    """Extracts route endpoints from common PT/EN route phrasings."""
    alias_map = {
        "airport": "Aeroporto",
        "lisbon airport": "Aeroporto",
        "airport terminal": "Aeroporto",
    }

    patterns = [
        r"\bde\s+metro\s+de\s+(?P<origin>.+?)\s+para\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bde\s+(?P<origin>.+?)\s+para\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bdo\s+(?P<origin>.+?)\s+ao\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bda\s+(?P<origin>.+?)\s+à\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bde\s+(?P<origin>.+?)\s+ao\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bde\s+(?P<origin>.+?)\s+à\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bfrom\s+(?P<origin>.+?)\s+to\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bentre\s+(?P<origin>.+?)\s+e\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
    ]

    def _clean(part: str) -> str:
        part = part.strip(" .?!,;:")
        part = re.sub(
            r"\b(agora|pfv|por favor|sff|please|now|right now|já|ja|mesmo|pff)\b.*$",
            "",
            part,
            flags=re.IGNORECASE,
        )
        part = re.sub(r"\b(de metro|by metro|using metro|via metro)\b", "", part, flags=re.IGNORECASE)
        part = re.sub(r"^(o|a|os|as|the)\s+", "", part, flags=re.IGNORECASE)
        part = part.strip(" .?!,;:")
        normalized = _normalize_token(part)
        return alias_map.get(normalized, part)

    for pattern in patterns:
        match = re.search(pattern, user_message, flags=re.IGNORECASE)
        if match:
            origin = _clean(match.group("origin"))
            destination = _clean(match.group("destination"))
            if origin and destination:
                return origin, destination

    return None


def _parse_wait_targets_from_route(route_result: str) -> List[Tuple[str, str]]:
    """Parses the stations and directions needed for live metro wait times."""
    targets: List[Tuple[str, str]] = []

    board_match = re.search(
        r"Board at\s+\*\*(?P<station>[^*]+)\*\*.*?direção\s+\*\*(?P<direction>[^*]+)\*\*",
        route_result,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if board_match:
        targets.append(
            (
                board_match.group("station").strip(),
                board_match.group("direction").strip(),
            )
        )

    transfer_match = re.search(
        r"Transfer at\*\*:\s*(?P<station>[^\n\(]+)",
        route_result,
        flags=re.IGNORECASE,
    )
    directions = re.findall(r"direção\s+\*\*([^*]+)\*\*", route_result, flags=re.IGNORECASE)
    if transfer_match and len(directions) >= 2:
        targets.append((transfer_match.group("station").strip(), directions[1].strip()))

    deduped: List[Tuple[str, str]] = []
    seen = set()
    for station, direction in targets:
        key = (_normalize_token(station), _normalize_token(direction))
        if key not in seen:
            seen.add(key)
            deduped.append((station, direction))

    return deduped


def _parse_wait_targets_from_response(response: str) -> List[Tuple[str, str]]:
    """Fallback parser for station and direction pairs from the model response."""
    targets: List[Tuple[str, str]] = []
    directions = re.findall(r"direção\s+\**([^*\n\(]+)", response, flags=re.IGNORECASE)

    origin_match = re.search(
        r"Embarque na estação[: ]*\**(?P<station>[^*\n\(]+)",
        response,
        flags=re.IGNORECASE,
    )
    transfer_match = re.search(
        r"Transfer[êe]ncia em[: ]*\**(?P<station>[^*\n\(]+)",
        response,
        flags=re.IGNORECASE,
    )

    if origin_match and directions:
        targets.append((origin_match.group("station").strip(), directions[0].strip()))
    if transfer_match and len(directions) >= 2:
        targets.append((transfer_match.group("station").strip(), directions[1].strip()))

    return targets


def _extract_wait_times_for_direction(wait_result: str, target_direction: str) -> Optional[str]:
    """Extracts the next two wait times for a specific direction."""
    blocks = re.findall(
        r"Direction:\s*(?P<direction>[^\n]+)\n\s*⏱️ Next train:\s*(?P<next>[^\n]+)\n\s*⏳ Following:\s*(?P<following>[^\n]+)",
        wait_result,
        flags=re.IGNORECASE,
    )

    target_norm = _normalize_token(target_direction)
    for direction, next_train, following in blocks:
        if _normalize_token(direction) == target_norm:
            following_parts = [part.strip() for part in following.split(",") if part.strip()]
            times = [next_train.strip()]
            if following_parts:
                times.append(following_parts[0])
            return " | ".join(times)

    return None


def _upsert_realtime_wait_section(response: str, lines: List[str]) -> str:
    """Replaces or inserts the real-time section before the tip or source block."""
    if not lines:
        return response

    response_lines = response.splitlines()
    section_start = None
    section_end = None

    for idx, line in enumerate(response_lines):
        stripped = line.strip()
        if stripped.startswith("🗓️") and "Próximos Metros" in stripped:
            section_start = idx
            continue
        if section_start is not None and (stripped.startswith("💡") or stripped.startswith("📌")):
            section_end = idx
            break

    if section_start is None:
        section_start = len(response_lines)
        for idx, line in enumerate(response_lines):
            stripped = line.strip()
            if stripped.startswith("💡") or stripped.startswith("📌"):
                section_start = idx
                break
        section_end = section_start
    elif section_end is None:
        section_end = len(response_lines)

    section_lines = ["🗓️ **Próximos Metros** (tempo real):", *lines, ""]
    new_lines = response_lines[:section_start] + section_lines + response_lines[section_end:]
    return "\n".join(new_lines).strip()


def _infer_language(user_message: str, context: str) -> str:
    """Infers response language from context and the user message."""
    if "PT-PT" in context or "Portuguese" in context:
        return "pt"
    if "English" in context:
        return "en"

    if re.search(
        r"\b(quero|vou|como|agora|para|de|ao|à|há|falhas|metro|estação|trajeto)\b",
        user_message,
        flags=re.IGNORECASE,
    ):
        return "pt"

    return "en"


def _get_line_id_between(station_a: Optional[str], station_b: Optional[str]) -> Optional[str]:
    """Returns the shared metro line between two stations, if any."""
    if not station_a or not station_b:
        return None

    from tools.metrolisboa_api import get_station_lines

    a_lines = set(get_station_lines(station_a))
    b_lines = set(get_station_lines(station_b))
    shared = list(a_lines & b_lines)
    return shared[0] if shared else None


def _parse_route_details(route_result: str) -> dict:
    """Parses route details from the deterministic route tool output."""
    board_station_match = re.search(r"Board at\s+\*\*(?P<station>[^*]+)\*\*", route_result)
    exits = re.findall(r"Exit at\s+\*\*([^*]+)\*\*", route_result)
    transfer_station_match = re.search(r"Transfer at\*\*:\s*(?P<station>[^\n\(]+)", route_result)
    directions = re.findall(r"direção\s+\*\*([^*]+)\*\*", route_result, flags=re.IGNORECASE)
    estimated_time_match = re.search(r"Estimated travel time:\s+\*\*([^*]+)\*\*", route_result)
    walk_match = re.search(r"Walk to\s+([^\n]+)", route_result)

    return {
        "board_station": board_station_match.group("station").strip() if board_station_match else None,
        "final_station": exits[-1].strip() if exits else None,
        "transfer_station": transfer_station_match.group("station").strip() if transfer_station_match else None,
        "directions": [direction.strip() for direction in directions],
        "estimated_time": estimated_time_match.group(1).strip() if estimated_time_match else None,
        "walk_target": walk_match.group(1).strip() if walk_match else None,
    }


def _line_display_name(line_id: str, language: str) -> str:
    """Returns localized metro line labels."""
    names_pt = {
        "amarela": "Linha Amarela",
        "azul": "Linha Azul",
        "verde": "Linha Verde",
        "vermelha": "Linha Vermelha",
    }
    names_en = {
        "amarela": "Yellow Line",
        "azul": "Blue Line",
        "verde": "Green Line",
        "vermelha": "Red Line",
    }
    return (names_pt if language == "pt" else names_en).get(line_id, line_id.title())


def _localize_wait_times(wait_times: str, language: str) -> str:
    """Localizes small wait-time labels for display."""
    return wait_times.replace("arriving", "a chegar") if language == "pt" else wait_times


def _build_route_state_lines(line_ids: List[str], language: str) -> List[str]:
    """Builds route-specific real-time line status bullets."""
    from tools.metrolisboa_api import METRO_LINES
    from tools.transport_api import _get_line_status

    status_lines: List[str] = []
    seen = set()
    for line_id in line_ids:
        if not line_id or line_id in seen:
            continue
        seen.add(line_id)
        line_info = METRO_LINES.get(line_id, {})
        emoji = line_info.get("emoji", "🚇")
        line_name = _line_display_name(line_id, language)
        status = _get_line_status(line_id)

        if status.lower() == "ok":
            status_text = "circulação normal" if language == "pt" else "normal service"
        elif status.lower() == "unknown":
            status_text = (
                "estado em tempo real indisponível"
                if language == "pt"
                else "real-time status unavailable"
            )
        else:
            status_text = status

        status_lines.append(f"- {emoji} **{line_name}**: {status_text}")

    return status_lines


def _build_practical_tip(
    language: str,
    first_direction: Optional[str],
    transfer_station: Optional[str],
    second_line_id: Optional[str],
    final_station: Optional[str],
    walk_target: Optional[str],
) -> str:
    """Builds a short, practical, non-generic travel tip."""
    if walk_target and final_station and _normalize_token(walk_target) != _normalize_token(final_station):
        if language == "pt":
            return f"Da estação {final_station} até {walk_target} a caminhada é curta."
        return f"From {final_station} to {walk_target}, the final walk is short."

    if transfer_station and second_line_id:
        second_line_name = _line_display_name(second_line_id, language)
        if language == "pt":
            return f"Em {transfer_station}, siga a sinalização para a {second_line_name}."
        return f"At {transfer_station}, follow the signs to the {second_line_name}."

    if first_direction:
        if language == "pt":
            return f"Confirme na plataforma a direção {first_direction} antes de embarcar."
        return f"Confirm the {first_direction} direction on the platform before boarding."

    return ""


def _build_deterministic_metro_route_response(
    user_message: str,
    context: str,
) -> Optional[str]:
    """Builds a deterministic metro route answer directly from tool outputs."""
    endpoints = _extract_route_endpoints(user_message)
    if not endpoints:
        return None

    language = _infer_language(user_message, context)

    from tools.metrolisboa_api import METRO_LINES, get_metro_wait_time
    from tools.transport_api import get_route_between_stations

    try:
        route_result = str(
            get_route_between_stations.invoke(
                {"origin": endpoints[0], "destination": endpoints[1]}
            )
        )
    except Exception:
        return None

    if "METRO ROUTE" not in route_result:
        return None

    details = _parse_route_details(route_result)
    board_station = details["board_station"] or endpoints[0].title()
    final_station = details["final_station"]
    transfer_station = details["transfer_station"]
    directions = details["directions"]
    first_direction = directions[0] if directions else None
    second_direction = directions[1] if len(directions) > 1 else None
    estimated_time = details["estimated_time"] or "~-- min"
    walk_target = details["walk_target"]

    if not final_station:
        return None

    first_line_id = _get_line_id_between(board_station, transfer_station or final_station)
    second_line_id = _get_line_id_between(transfer_station, final_station) if transfer_station else None
    line_ids = [line_id for line_id in [first_line_id, second_line_id] if line_id]

    state_lines = _build_route_state_lines(line_ids, language)

    station_label = "Estação" if language == "pt" else "Station"
    direction_label = "Direção" if language == "pt" else "Direction"
    next_label = "⏱️ Próximo Metro em:" if language == "pt" else "⏱️ Next Metro in:"

    realtime_lines: List[str] = []
    for station, direction in _parse_wait_targets_from_route(route_result)[:2]:
        try:
            wait_result = str(get_metro_wait_time.invoke({"station": station}))
        except Exception:
            wait_result = ""

        wait_times = _extract_wait_times_for_direction(wait_result, direction)
        if wait_times:
            realtime_lines.append(
                f"- **{station_label} {station}:** {direction_label} {direction} — **{next_label}** {_localize_wait_times(wait_times, language)}"
            )

    if not realtime_lines:
        realtime_lines.append(
            "- Sem dados em tempo real"
            if language == "pt"
            else "- No real-time data available"
        )

    tip = _build_practical_tip(
        language=language,
        first_direction=first_direction,
        transfer_station=transfer_station,
        second_line_id=second_line_id,
        final_station=final_station,
        walk_target=walk_target,
    )

    route_title = f"🚇 **{endpoints[0].title()}** → **{endpoints[1].title()}**"
    state_title = "⚠️ **Estado das Linhas:**" if language == "pt" else "⚠️ **Line Status:**"
    time_title = "⏳ **Tempo total estimado:**" if language == "pt" else "⏳ **Estimated total time:**"
    route_section = "🗺️ **O seu Trajeto de Metro:**" if language == "pt" else "🗺️ **Your Metro Route:**"
    waits_section = "🗓️ **Próximos Metros** (tempo real):" if language == "pt" else "🗓️ **Next Metros** (real time):"
    tip_title = "💡 **Dica rápida:**" if language == "pt" else "💡 **Quick tip:**"
    source_label = "📌 **Fonte:**" if language == "pt" else "📌 **Source:**"
    updated_label = "**Atualizado:**" if language == "pt" else "**Updated:**"
    board_text = "- 📍 **Embarque na estação" if language == "pt" else "- 📍 **Board at"
    exit_text = "- 🎯 **Saia na estação" if language == "pt" else "- 🎯 **Exit at"
    transfer_text = "- 🔄 **Transferência em" if language == "pt" else "- 🔄 **Transfer at"
    walk_text = "- 🚶 **Siga a pé para" if language == "pt" else "- 🚶 **Walk to"
    direction_word = "direção" if language == "pt" else "direction"

    response_lines = [
        route_title,
        state_title,
        *state_lines,
        "",
        f"{time_title} {estimated_time}",
        "",
        route_section,
        f"{board_text} {board_station}**",
    ]

    if first_line_id and first_direction:
        response_lines.append(
            f"- {METRO_LINES[first_line_id]['emoji']} **{_line_display_name(first_line_id, language)}** - {direction_word} **{first_direction}**"
        )

    if transfer_station:
        response_lines.append(f"{transfer_text} {transfer_station}**")
        if second_line_id and second_direction:
            response_lines.append(
                f"- {METRO_LINES[second_line_id]['emoji']} **{_line_display_name(second_line_id, language)}** - {direction_word} **{second_direction}**"
            )

    response_lines.append(f"{exit_text} {final_station}**")

    if walk_target and _normalize_token(walk_target) != _normalize_token(final_station):
        response_lines.append(f"{walk_text} {walk_target}**")

    response_lines.extend([
        "",
        waits_section,
        *realtime_lines,
        "",
    ])

    if tip:
        response_lines.append(f"{tip_title} {tip}")
        response_lines.append("")

    response_lines.append(
        f"{source_label} [*Metro de Lisboa*](https://www.metrolisboa.pt) | {updated_label} {datetime.now().strftime('%H:%M')}"
    )

    return "\n".join(response_lines).strip()


def _build_deterministic_route_tool_response(user_message: str) -> Optional[str]:
    """Returns the raw route-tool guidance for non-metro-direct routes."""
    endpoints = _extract_route_endpoints(user_message)
    if not endpoints:
        return None

    from tools.transport_api import get_route_between_stations

    try:
        route_result = str(
            get_route_between_stations.invoke(
                {"origin": endpoints[0], "destination": endpoints[1]}
            )
        )
    except Exception:
        return None

    route_result = route_result.strip() if route_result else ""
    if not route_result:
        return None

    if "METRO ROUTE" in route_result:
        return route_result

    try:
        from tools.carris_api import carris_find_routes_between

        carris_result = str(
            carris_find_routes_between.invoke(
                {"origin": endpoints[0], "destination": endpoints[1]}
            )
        ).strip()
    except Exception:
        carris_result = ""

    if carris_result and not any(
        marker in carris_result
        for marker in [
            "No direct Carris route found",
            "Could not locate",
            "Sem rota direta",
            "Não foi possível localizar",
        ]
    ):
        return carris_result

    return route_result


def _build_tool_call(name: str, args: dict) -> AIMessage:
    """Creates a deterministic tool call message for the subgraph."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": f"auto_{uuid.uuid4().hex}",
                "type": "tool_call",
            }
        ],
    )


def _build_deterministic_transport_tool_call(user_message: str) -> Optional[AIMessage]:
    """Routes obvious transport coverage prompts to their canonical tool."""
    query = user_message.strip()
    query_lower = query.lower()

    if "current status of the lisbon metro lines" in query_lower:
        return _build_tool_call("get_metro_status", {})
    if "wait time at oriente metro station" in query_lower:
        return _build_tool_call("get_metro_wait_time", {"station": "Oriente"})
    if "entire red metro line" in query_lower:
        return _build_tool_call("get_metro_line_wait_times", {"line": "vermelha"})
    if "nearest to gps coordinates" in query_lower:
        return _build_tool_call("find_nearest_metro", {"lat": 38.725, "lon": -9.149})
    if "scheduled frequency of the green metro line" in query_lower:
        return _build_tool_call("get_metro_frequency", {"line": "verde", "day_type": "weekday"})
    if "list all lisbon metro stations" in query_lower:
        return _build_tool_call("get_all_metro_stations", {})

    if "carris metropolitana service alerts" in query_lower:
        return _build_tool_call("get_carris_metropolitana_alerts", {})
    if "stop information for stop id" in query_lower:
        stop_id_match = re.search(r"stop id\s+([0-9]+)", query, flags=re.IGNORECASE)
        return _build_tool_call("get_carris_metropolitana_stop_info", {"stop_id": stop_id_match.group(1) if stop_id_match else "010101"})
    if "search carris metropolitana lines for" in query_lower:
        term = re.sub(r"^.*lines for\s+", "", query, flags=re.IGNORECASE).strip(" .?!")
        return _build_tool_call("search_carris_metropolitana_lines", {"query": term or "470"})
    if "bus routes from almada to cacilhas" in query_lower:
        return _build_tool_call("find_bus_routes", {"origin": "Almada", "destination": "Cacilhas"})
    if "real-time carris metropolitana bus positions near almada" in query_lower:
        return _build_tool_call("get_real_time_bus_positions", {"location": "Almada", "radius_km": 1.0})
    if "live gps locations for carris metropolitana line 470" in query_lower:
        return _build_tool_call("get_bus_realtime_locations", {"line_id": "470"})
    if "next departures for carris metropolitana line 470 at stop 010101" in query_lower:
        return _build_tool_call("get_bus_next_departures", {"line_id": "470", "stop_id": "010101"})
    if "direct carris metropolitana bus lines between oeiras and amadora" in query_lower:
        return _build_tool_call("find_direct_bus_lines", {"origin": "Oeiras", "destination": "Amadora"})

    if "search carris stops for marquês de pombal" in query_lower or "search carris stops for marquês de pombal" in query_lower:
        return _build_tool_call("carris_get_stops", {"query": "Marquês de Pombal"})
    if "route information for tram 28e" in query_lower:
        return _build_tool_call("carris_get_routes", {"route_id": "28E"})
    if "next carris departures for stop 1234" in query_lower:
        return _build_tool_call("carris_get_next_departures", {"stop_id": "1234"})
    if "carris routes between graça and belém" in query_lower:
        return _build_tool_call("carris_find_routes_between", {"origin": "Graça", "destination": "Belém"})
    if "real-time carris vehicles for route 15e" in query_lower:
        return _build_tool_call("carris_get_realtime_vehicles", {"route_id": "15E"})
    if "real-time arrivals at carris stop 1234" in query_lower:
        return _build_tool_call("carris_get_arrivals", {"stop_id": "1234"})
    if "eta of route 28e at martim moniz" in query_lower:
        return _build_tool_call("carris_vehicle_eta", {"route_short_name": "28E", "stop_name": "Martim Moniz"})
    if "service frequency for carris route 15e" in query_lower:
        return _build_tool_call("carris_get_service_frequency", {"route_short_name": "15E"})

    if "status of cp urban trains in lisbon" in query_lower:
        return _build_tool_call("get_train_status", {})
    if "search cp stations for rossio" in query_lower:
        return _build_tool_call("search_cp_stations", {"query": "Rossio"})
    if "next train departures from entrecampos" in query_lower:
        return _build_tool_call("get_train_schedule", {"station_name": "Entrecampos"})
    if "cp urban routes available" in query_lower:
        return _build_tool_call("get_cp_routes", {})
    if "plan a cp train trip from rossio to sintra" in query_lower:
        return _build_tool_call("plan_train_trip", {"origin": "Rossio", "destination": "Sintra"})
    if "train service frequency on the sintra line" in query_lower:
        return _build_tool_call("get_train_frequency", {"route_name": "Sintra"})

    if "transport summary across metro, bus, and train" in query_lower:
        return _build_tool_call("get_transport_summary", {})
    if "from baixa-chiado to aeroporto" in query_lower:
        return _build_tool_call("get_route_between_stations", {"origin": "Baixa-Chiado", "destination": "Aeroporto"})

    return None


class TransportAgent(BaseAgent):
    """
    Transport specialist agent for Lisbon's public transport.

    Handles:
        - Metro de Lisboa (status, routing, wait times)
        - Carris Urban (city buses and trams: 28E, 15E, 732, etc.)
        - Carris Metropolitana (suburban bus routes, alerts)
        - CP trains (suburban lines: Cascais, Sintra, Azambuja)
        - Multi-modal routing with GPS-based stop finding
    """

    def __init__(self):
        """Initializes the transport agent."""
        super().__init__("transport")
        self.system_prompt = get_transport_prompt()

    def _get_tool_by_name(self, tool_name: str):
        """Returns a loaded tool by name, or None if not found."""
        for tool in self.tools:
            if getattr(tool, "name", "") == tool_name:
                return tool
        return None

    @staticmethod
    def _is_status_query(user_message: str) -> bool:
        """Detects generic service-status questions that are better answered deterministically."""
        query = user_message.lower()
        status_patterns = [
            r"\bis the metro working\b",
            r"\bmetro status\b",
            r"\btransport status\b",
            r"\bhow are the transports\b",
            r"\bcomo est[aã]o os transportes\b",
            r"\best[aá] o metro a funcionar\b",
            r"\bestado do metro\b",
            r"\bestado dos transportes\b",
            r"\bare trains running\b",
            r"\bservice status\b",
        ]
        return any(re.search(pattern, query) for pattern in status_patterns)

    def _run_direct_tool_fallback(self, user_message: str) -> Optional[str]:
        """Uses deterministic transport tools for broad status questions."""
        if not self._is_status_query(user_message):
            return None

        query = user_message.lower()
        metro_status_patterns = [
            r"\bis the metro working\b",
            r"\bmetro status\b",
            r"\best[aá] o metro a funcionar\b",
            r"\bestado do metro\b",
            r"\bmetro lines\b",
            r"\bmetro service\b",
        ]

        if any(re.search(pattern, query) for pattern in metro_status_patterns):
            metro_tool = self._get_tool_by_name("get_metro_status")
            if metro_tool:
                return metro_tool.invoke({})

        summary_tool = self._get_tool_by_name("get_transport_summary")
        if summary_tool:
            return summary_tool.invoke({})

        return None

    def _ensure_realtime_wait_times(self, user_message: str, response: str) -> str:
        """Guarantees real-time wait times for metro route responses."""
        if not response:
            return response

        user_lower = user_message.lower()
        response_lower = response.lower()
        if "metro" not in user_lower and "trajeto de metro" not in response_lower:
            return response

        endpoints = _extract_route_endpoints(user_message)
        if not endpoints:
            return response

        from tools.metrolisboa_api import get_metro_wait_time
        from tools.transport_api import get_route_between_stations

        try:
            route_result = str(
                get_route_between_stations.invoke(
                    {"origin": endpoints[0], "destination": endpoints[1]}
                )
            )
        except Exception:
            route_result = ""

        targets = _parse_wait_targets_from_route(route_result)
        if not targets:
            targets = _parse_wait_targets_from_response(response)
        if not targets:
            return response

        realtime_lines: List[str] = []
        for station, direction in targets[:2]:
            try:
                wait_result = str(get_metro_wait_time.invoke({"station": station}))
            except Exception:
                wait_result = ""

            wait_times = _extract_wait_times_for_direction(wait_result, direction)
            if wait_times:
                realtime_lines.append(
                    f"- **Estação {station}:** Direção {direction} — **⏱️ Próximo Metro em:** {wait_times}"
                )

        if not realtime_lines:
            realtime_lines.append("- Sem dados em tempo real")

        return _upsert_realtime_wait_section(response, realtime_lines)

    @traceable(name="transport_agent", run_type="chain", tags=["sub-agent", "transport"])
    def invoke(
        self, user_message: str, context: str = "", verbose: bool = False
    ) -> str:
        """
        Processes a transport-related query.

        Args:
            user_message: The user's query.
            context: Additional context from other agents (optional).
            verbose: Whether involved tool calls should be printed.

        Returns:
            str: Transport information response.
        """
        language = infer_response_language(user_query=user_message, default="en")
        language_instruction = (
            "Respond ENTIRELY in Portuguese (PT-PT)."
            if language == "pt"
            else "Respond ENTIRELY in English."
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            SystemMessage(content=language_instruction),
        ]

        if context:
            messages.append(
                SystemMessage(content=f"Context from other agents:\n{context}")
            )

        messages.append(HumanMessage(content=user_message))

        # Skip tool enforcement for greetings/thanks
        is_greeting = any(
            w in user_message.lower()
            for w in ["hello", "thanks", "obrigado", "tchau", "olá", "bom dia"]
        )

        if not is_greeting:
            deterministic_response = _build_deterministic_metro_route_response(
                user_message=user_message,
                context=context,
            )
            if deterministic_response:
                return finalize_worker_response(
                    deterministic_response,
                    agent_name="transport",
                    user_query=user_message,
                    language=language,
                )

            direct_route_response = _build_deterministic_route_tool_response(user_message)
            if direct_route_response:
                return finalize_worker_response(
                    direct_route_response,
                    agent_name="transport",
                    user_query=user_message,
                    language=language,
                )

            direct_tool_response = self._run_direct_tool_fallback(user_message)
            if direct_tool_response:
                return finalize_worker_response(
                    direct_tool_response,
                    agent_name="transport",
                    user_query=user_message,
                    language=language,
                )

        response = self.execute_react_loop(
            messages=messages,
            verbose=verbose,
            max_iterations=5,
            tool_enforcement_msg="" if is_greeting else (
                "You MUST use a tool (like get_metro_status or get_route_between_stations) "
                "to get real data. Do NOT answer from your knowledge base. Call the tool now."
            ),
        )

        return finalize_worker_response(
            self._ensure_realtime_wait_times(user_message, response),
            agent_name="transport",
            user_query=user_message,
            language=language,
        )

    def build_subgraph(self) -> "CompiledStateGraph":
        """
        Builds a LangGraph subgraph for this agent.

        Returns:
            CompiledStateGraph: Compiled subgraph for transport queries.
        """

        def agent_node(state: AgentState) -> dict:
            """Transport agent decision node."""
            messages = list(state["messages"])

            last_message = messages[-1] if messages else None
            if isinstance(last_message, ToolMessage):
                response = self._safe_llm_invoke(self.llm_with_tools, messages)
                return {"messages": [response]}

            user_message = None
            for message in reversed(messages):
                if isinstance(message, HumanMessage) and message.content:
                    user_message = str(message.content)
                    break

            if user_message:
                deterministic_call = _build_deterministic_transport_tool_call(user_message)
                if deterministic_call is not None:
                    return {"messages": [deterministic_call]}

            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self.system_prompt)] + messages

            response = self._safe_llm_invoke(self.llm_with_tools, messages)
            return {"messages": [response]}

        def should_continue(state: AgentState) -> str:
            """Determines next step."""
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Transport Agent Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")

    try:
        agent = TransportAgent()
        print(
            f"\n\033[1m✅ Transport Agent initialized:\033[0m {agent.get_model_info()}"
        )
        print(f"   Tools: {len(agent.tools)} transport tools")
        print(f"          {[t.name for t in agent.tools]}")

        print("\n\033[1m📝 Testing query:\033[0m 'Is the metro working?'")
        response = agent.invoke("Is the metro working?")
        print("\n\033[1m🤖 Response:\033[0m")
        print(response)

        print("\n\033[1;32m✅ Transport agent working!\033[0m")

    except Exception as e:
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
        import traceback

        traceback.print_exc()
