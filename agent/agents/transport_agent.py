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
from difflib import SequenceMatcher
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from agent.agents.base import BaseAgent, traceable
from agent.prompts.transport import get_transport_prompt
from agent.state import AgentState
from agent.utils.langgraph_compat import ToolNode
from agent.utils.response_formatter import (
    extract_update_time,
    finalize_worker_response,
    infer_response_language,
)


def _normalize_token(text: str) -> str:
    """Normalizes station and direction tokens for robust matching."""
    import unicodedata

    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    return normalized.lower().strip()


def _clean_query_fragment(part: str) -> str:
    """Cleans station and stop fragments parsed from transport questions."""
    alias_map = {
        "airport": "Aeroporto",
        "lisbon airport": "Aeroporto",
        "airport terminal": "Aeroporto",
    }

    part = part.strip(" .?!,;:")
    part = re.sub(
        r"\b(agora|pfv|por favor|sff|please|now|right now|já|ja|mesmo|pff)\b.*$",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"\b(de metro|by metro|using metro|via metro|by bus|by tram|by train|de autocarro|de comboio|de elétrico|de eletrico)\b",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"\b(metro station|station|esta[cç][aã]o|stop|paragem|terminal)\b",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(r"^(o|a|os|as|the)\s+", "", part, flags=re.IGNORECASE)
    part = part.strip(" .?!,;:")
    normalized = _normalize_token(part)
    return alias_map.get(normalized, part)


def _extract_coordinates(query: str) -> Optional[Tuple[float, float]]:
    """Extracts latitude/longitude pairs from a transport query when available."""
    match = re.search(
        r"(?P<latitude>-?\d{1,2}\.\d+)\s*[,;/]\s*(?P<longitude>-?\d{1,3}\.\d+)",
        query,
    )
    if not match:
        return None

    try:
        return float(match.group("latitude")), float(match.group("longitude"))
    except (TypeError, ValueError):
        return None


def _canonicalize_route_code(raw_route: str) -> Optional[str]:
    """Normalizes Carris route codes such as `28 E` -> `28E`."""
    cleaned = re.sub(r"\s+", "", str(raw_route or "").upper())
    if re.fullmatch(r"\d{1,4}[A-Z]?", cleaned):
        return cleaned
    return None


def _extract_route_code(query: str) -> Optional[str]:
    """Extracts a Carris bus/tram route code from a natural-language query."""
    patterns = [
        r"\b(?:route|linha|line|tram|trams|el[eé]trico|eletrico|bus|buses|autocarro|autocarros)\s*(?:number|n[uú]mero|n[oº]?|#)?\s*(?P<route>\d{1,4}\s*[A-Za-z]?)\b",
        r"\b(?P<route>\d{1,4}\s*E)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            route_code = _canonicalize_route_code(match.group("route"))
            if route_code:
                return route_code
    return None


def _resolve_named_candidate(
    fragment: str,
    candidate_map: Dict[str, str],
    minimum_score: float = 0.78,
) -> Optional[str]:
    """Resolves a free-form place name against a candidate map with fuzzy matching."""
    normalized_fragment = _normalize_token(fragment)
    if not normalized_fragment:
        return None

    if normalized_fragment in candidate_map:
        return candidate_map[normalized_fragment]

    best_value = None
    best_score = 0.0
    fragment_tokens = set(normalized_fragment.split())
    for candidate_key, canonical in candidate_map.items():
        candidate_tokens = set(candidate_key.split())
        score = SequenceMatcher(None, normalized_fragment, candidate_key).ratio()
        if normalized_fragment in candidate_key or candidate_key in normalized_fragment:
            score += 0.12
        if fragment_tokens and candidate_tokens:
            overlap = len(fragment_tokens & candidate_tokens) / max(
                len(fragment_tokens), len(candidate_tokens)
            )
            score += overlap * 0.25
        if score > best_score:
            best_score = score
            best_value = canonical

    return best_value if best_score >= minimum_score else None


@lru_cache(maxsize=1)
def _get_metro_station_name_map() -> Dict[str, str]:
    """Builds a normalized alias map for Metro station names."""
    from tools.metrolisboa_api import METRO_STATION_IDS, load_metro_stations

    stations_by_code: Dict[str, str] = {}
    for station in load_metro_stations():
        stop_id = str(station.get("stop_id", "")).strip()
        stop_name = str(station.get("stop_name", "")).strip()
        if stop_id and stop_name:
            stations_by_code[stop_id] = stop_name

    alias_map: Dict[str, str] = {}
    for alias, station_id in METRO_STATION_IDS.items():
        canonical = stations_by_code.get(station_id) or alias.title()
        alias_map[_normalize_token(alias)] = canonical

    for canonical in stations_by_code.values():
        alias_map[_normalize_token(canonical)] = canonical

    return alias_map


def _resolve_metro_station_name(fragment: str) -> str:
    """Resolves a Metro station fragment to the best canonical station name."""
    cleaned = _clean_query_fragment(fragment)
    resolved = _resolve_named_candidate(cleaned, _get_metro_station_name_map())
    return resolved or cleaned


@lru_cache(maxsize=1)
def _get_cp_station_name_map() -> Dict[str, str]:
    """Builds a normalized alias map for CP AML station names."""
    from tools.cp_api import CP_KEY_STATIONS

    alias_map: Dict[str, str] = {}
    for key, info in CP_KEY_STATIONS.items():
        canonical = str(info.get("name") or key.replace("_", " ").title()).strip()
        alias_map[_normalize_token(canonical)] = canonical
        alias_map[_normalize_token(key.replace("_", " "))] = canonical

    extra_aliases = {
        "cais do sodre": "Cais do Sodré",
        "lisboa oriente": "Lisboa - Oriente",
        "santa apolonia": "Santa Apolónia",
        "belem": "Belém",
    }
    for alias, canonical in extra_aliases.items():
        alias_map[_normalize_token(alias)] = canonical

    return alias_map


def _resolve_cp_station_name(fragment: str) -> str:
    """Resolves a CP station fragment to the best canonical AML station name."""
    cleaned = _clean_query_fragment(fragment)
    resolved = _resolve_named_candidate(cleaned, _get_cp_station_name_map(), minimum_score=0.74)
    return resolved or cleaned


def _extract_metro_line_id(query: str) -> Optional[str]:
    """Extracts the canonical Lisbon Metro line ID from a user query."""
    normalized = _normalize_token(query)
    aliases = {
        "yellow line": "amarela",
        "linha amarela": "amarela",
        "amarela": "amarela",
        "yellow": "amarela",
        "blue line": "azul",
        "linha azul": "azul",
        "azul": "azul",
        "blue": "azul",
        "green line": "verde",
        "linha verde": "verde",
        "verde": "verde",
        "green": "verde",
        "red line": "vermelha",
        "linha vermelha": "vermelha",
        "vermelha": "vermelha",
        "red": "vermelha",
    }
    for alias, line_id in aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return line_id
    return None


def _build_metro_tool_spec(user_message: str) -> Optional[Dict[str, Any]]:
    """Maps natural-language Lisbon Metro queries to deterministic tool specs."""
    query = user_message.strip()
    query_lower = query.lower()
    line_id = _extract_metro_line_id(query)
    has_metro_context = "metro" in query_lower or line_id is not None

    if has_metro_context and not _query_has_wait_departure_intent(query) and (
        _query_has_status_intent(query)
        or re.search(r"\b(status|estado|working|a funcionar|service)\b", query_lower)
    ):
        return {"name": "get_metro_status", "args": {}}

    if has_metro_context and re.search(
        r"\b(all|list|show|every|todas?|listar|quais)\b.*\b(stations|esta[cç][õo]es|estacoes)\b|\b(stations|esta[cç][õo]es|estacoes)\b.*\bmetro\b",
        query_lower,
    ):
        return {"name": "get_all_metro_stations", "args": {}}

    if line_id and re.search(
        r"\b(wait times?|tempos? de espera|entire|whole|all stations|linha toda|toda a linha)\b",
        query_lower,
    ):
        return {"name": "get_metro_line_wait_times", "args": {"line": line_id}}

    if line_id and re.search(
        r"\b(frequency|headway|how often|interval|intervalo|frequ[eê]ncia|de quanto em quanto)\b",
        query_lower,
    ):
        return {
            "name": "get_metro_frequency",
            "args": {"line": line_id, "day_type": "weekday"},
        }

    if "metro" in query_lower and re.search(
        r"\b(nearest|closest|mais perto|mais próxima|mais proxima)\b",
        query_lower,
    ):
        coordinates = _extract_coordinates(query)
        if coordinates:
            return {
                "name": "find_nearest_metro",
                "args": {"latitude": coordinates[0], "longitude": coordinates[1]},
            }

        near_match = re.search(
            r"\b(?:near|to|for|perto de|junto de|ao p[eé] de)\s+(?P<location>.+?)(?:[\?\!\.,;]|$)",
            query,
            flags=re.IGNORECASE,
        )
        if near_match:
            return {
                "name": "find_nearest_metro",
                "args": {"near_location_name": _clean_query_fragment(near_match.group("location"))},
            }

    return None


def _build_carris_urban_tool_spec(user_message: str) -> Optional[Dict[str, Any]]:
    """Maps natural-language Carris urban bus/tram queries to deterministic tool specs."""
    query = user_message.strip()
    query_lower = query.lower()
    endpoints = _extract_route_endpoints(query)
    route_code = _extract_route_code(query)
    has_train_context = bool(re.search(r"\b(cp|comboio|comboios|train|trains)\b", query_lower))

    if has_train_context or _looks_like_carris_metropolitana_query(query, endpoints=endpoints):
        return None

    stop_id_match = re.search(r"\b(?:stop|paragem)\s+(?P<stop_id>\d{2,})\b", query_lower)

    stop_search_patterns = [
        r"(?:search|find|lookup|show)\s+(?:carris\s+)?stops?\s+(?:for|near|named)\s+(?P<term>.+?)(?:[\?\!\.,;]|$)",
        r"(?:paragens?|stops?)\s+(?:da\s+carris\s+)?(?:em|na|no|para|perto de)\s+(?P<term>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in stop_search_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {
                "name": "carris_get_stops",
                "args": {"query": _clean_query_fragment(match.group("term"))},
            }

    if stop_id_match and re.search(r"\b(arrivals|next arrivals|chegadas|pr[oó]ximas\s+chegadas)\b", query_lower):
        return {
            "name": "carris_get_arrivals",
            "args": {"stop_id": stop_id_match.group("stop_id")},
        }

    if stop_id_match and re.search(
        r"\b(next departures?|departures?|pr[oó]ximas?\s+partidas?)\b",
        query_lower,
    ):
        return {
            "name": "carris_get_next_departures",
            "args": {"stop_id": stop_id_match.group("stop_id")},
        }

    if route_code and re.search(
        r"\b(real-time|realtime|live|gps|location|locations|position|positions|where|onde|agora|now)\b",
        query_lower,
    ):
        return {
            "name": "carris_get_realtime_vehicles",
            "args": {"route_short_name": route_code},
        }

    if route_code and re.search(
        r"\b(frequency|headway|how often|interval|intervalo|frequ[eê]ncia|de quanto em quanto)\b",
        query_lower,
    ):
        return {
            "name": "carris_get_service_frequency",
            "args": {"route_short_name": route_code},
        }

    if route_code and re.search(
        r"\b(eta|estimated arrival|estimate the eta|quando chega|chega a que horas)\b",
        query_lower,
    ):
        stop_match = re.search(
            r"\b(?:at|na|no|em)\s+(?P<stop>.+?)(?:[\?\!\.,;]|$)",
            query,
            flags=re.IGNORECASE,
        )
        if stop_match:
            return {
                "name": "carris_vehicle_eta",
                "args": {
                    "route_short_name": route_code,
                    "stop_name": _clean_query_fragment(stop_match.group("stop")),
                },
            }

    if route_code and re.search(
        r"\b(route information|route info|show route|show .*details|details|detalhes?|rota)\b",
        query_lower,
    ):
        return {"name": "carris_get_routes", "args": {"route_id": route_code}}

    if endpoints and re.search(
        r"\b(carris|tram|trams|el[eé]trico|eletrico|bus|buses|autocarro|autocarros)\b",
        query_lower,
    ):
        return {
            "name": "carris_find_routes_between",
            "args": {"origin": endpoints[0], "destination": endpoints[1]},
        }

    return None


def _query_has_status_intent(query: str) -> bool:
    """Returns whether the query is primarily asking about service status."""
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
    return any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in status_patterns)


def _query_has_wait_departure_intent(query: str) -> bool:
    """Returns whether the query asks for next departures, arrivals, wait times, or ETAs."""
    patterns = [
        r"\bnext\b",
        r"\bwait time\b",
        r"\bwhen is the next\b",
        r"\bdeparture(?:s)?\b",
        r"\barrival(?:s)?\b",
        r"\beta\b",
        r"\bpr[oó]xim[oa]s?\b",
        r"\bchegada(?:s)?\b",
        r"\bpartida(?:s)?\b",
        r"\bquando passa\b",
        r"\bquanto tempo falta\b",
    ]
    return any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in patterns)


def _parse_metro_wait_blocks(wait_result: str) -> List[Dict[str, Any]]:
    """Parses structured wait-time blocks from Metro tool output."""
    blocks: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"Direction:\s*(?P<direction>[^\n]+)\n(?:\s*ℹ️\s*(?P<note>[^\n]+)\n)?\s*⏱️ Next train:\s*(?P<next>[^\n]+)\n\s*⏳ Following:\s*(?P<following>[^\n]+)",
        flags=re.IGNORECASE,
    )

    for match in pattern.finditer(wait_result or ""):
        following_parts = [
            part.strip() for part in match.group("following").split(",") if part.strip()
        ]
        blocks.append(
            {
                "direction": match.group("direction").strip(),
                "note": (match.group("note") or "").strip() or None,
                "times": [match.group("next").strip(), *following_parts],
            }
        )

    return blocks


def _extract_wait_block_for_direction(wait_result: str, target_direction: str) -> Optional[Dict[str, Any]]:
    """Extracts the full wait-time block for a requested metro direction."""
    target_norm = _normalize_token(target_direction)
    for block in _parse_metro_wait_blocks(wait_result):
        if _normalize_token(block["direction"]) == target_norm:
            return block
    return None


def _localize_platform_note(note: Optional[str], language: str) -> Optional[str]:
    """Localizes metro platform notes when possible."""
    if not note:
        return None
    if language != "pt":
        return note

    match = re.match(
        r"Platform indicator currently shows\s+(.+?)\.?$",
        note,
        flags=re.IGNORECASE,
    )
    if match:
        return f"O indicador de plataforma mostra de momento {match.group(1).strip()}."
    return note


def _build_metro_wait_lines(targets: List[Tuple[str, str]], language: str) -> List[str]:
    """Builds rich metro wait lines while preserving platform notes and all reported times."""
    from tools.metrolisboa_api import get_metro_wait_time

    station_label = "Estação" if language == "pt" else "Station"
    direction_label = "Direção" if language == "pt" else "Direction"
    next_label = "⏱️ Próximo Metro em:" if language == "pt" else "⏱️ Next Metro in:"

    realtime_lines: List[str] = []
    for station, direction in targets[:2]:
        try:
            wait_result = str(
                get_metro_wait_time.invoke(
                    {"station": station, "direction": direction}
                )
            )
        except Exception:
            wait_result = ""

        block = _extract_wait_block_for_direction(wait_result, direction)
        if not block:
            continue

        localized_times = [
            _localize_wait_times(time_text, language) for time_text in block.get("times", [])
        ]
        realtime_lines.append(
            f"- **{station_label} {station}:** {direction_label} {direction} — **{next_label}** {' | '.join(localized_times)}"
        )
        note = _localize_platform_note(block.get("note"), language)
        if note:
            realtime_lines.append(f"  ℹ️ {note}")

    if not realtime_lines:
        realtime_lines.append(
            "- Sem dados em tempo real" if language == "pt" else "- No real-time data available"
        )

    return realtime_lines


def _extract_route_endpoints(user_message: str) -> Optional[Tuple[str, str]]:
    """Extracts route endpoints from common PT/EN route phrasings."""
    patterns = [
        r"\bde\s+metro\s+de\s+(?P<origin>.+?)\s+para\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bde\s+(?P<origin>.+?)\s+para\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bdo\s+(?P<origin>.+?)\s+ao\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bda\s+(?P<origin>.+?)\s+à\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bde\s+(?P<origin>.+?)\s+ao\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bde\s+(?P<origin>.+?)\s+à\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bfrom\s+(?P<origin>.+?)\s+to\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bbetween\s+(?P<origin>.+?)\s+and\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bentre\s+(?P<origin>.+?)\s+e\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, user_message, flags=re.IGNORECASE)
        if match:
            origin = _clean_query_fragment(match.group("origin"))
            destination = _clean_query_fragment(match.group("destination"))
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
    block = _extract_wait_block_for_direction(wait_result, target_direction)
    if not block:
        return None

    times = list(block.get("times", []))[:2]
    return " | ".join(times) if times else None


def _upsert_realtime_wait_section(response: str, lines: List[str], language: str = "pt") -> str:
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

    section_title = "🗓️ **Próximos Metros** (tempo real):" if language == "pt" else "🗓️ **Next Metros** (real time):"
    section_lines = [section_title, *lines, ""]
    new_lines = response_lines[:section_start] + section_lines + response_lines[section_end:]
    return "\n".join(new_lines).strip()


def _infer_language(user_message: str, context: str) -> str:
    """Infers response language from context and the user message."""
    if "PT-PT" in context or "Portuguese" in context:
        return "pt"
    if "English" in context:
        return "en"

    return infer_response_language(user_query=user_message, default="en")


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


def _parse_metro_wait_request(user_message: str) -> Optional[Dict[str, Any]]:
    """Parses single-station metro wait queries with an optional direction."""
    query = user_message.strip()
    if "metro" not in query.lower():
        return None

    direction_patterns = [
        r"next metro at\s+(?P<station>.+?)\s+(?:towards|toward|to|direction)\s+(?P<direction>.+?)(?:[\?\!\.,;]|$)",
        r"when is the next metro at\s+(?P<station>.+?)\s+(?:towards|toward|to|direction)\s+(?P<direction>.+?)(?:[\?\!\.,;]|$)",
        r"pr[oó]ximo metro\s+(?:na|em)\s+(?P<station>.+?)\s+(?:sentido|dire[cç][aã]o)\s+(?P<direction>.+?)(?:[\?\!\.,;]|$)",
        r"quando passa o pr[oó]ximo metro\s+(?:na|em)\s+(?P<station>.+?)\s+(?:sentido|dire[cç][aã]o)\s+(?P<direction>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in direction_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {
                "station": _resolve_metro_station_name(match.group("station")),
                "direction": _resolve_metro_station_name(match.group("direction")),
                "status_requested": _query_has_status_intent(query),
            }

    station_only_patterns = [
        r"next metro at\s+(?P<station>.+?)(?:[\?\!\.,;]|$)",
        r"when is the next metro at\s+(?P<station>.+?)(?:[\?\!\.,;]|$)",
        r"metro wait time at\s+(?P<station>.+?)(?:[\?\!\.,;]|$)",
        r"pr[oó]ximo metro\s+(?:na|em)\s+(?P<station>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in station_only_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {
                "station": _resolve_metro_station_name(match.group("station")),
                "direction": None,
                "status_requested": _query_has_status_intent(query),
            }

    return None


def _resolve_carris_stop(stop_reference: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolves a Carris stop name into a stop ID and canonical stop label."""
    from tools.carris_api import _get_db_connection, _search_stop_rows

    conn = _get_db_connection()
    if not conn:
        return None, None

    try:
        rows = _search_stop_rows(conn, stop_reference, limit=5)
        if not rows:
            return None, None
        return rows[0]["stop_id"], rows[0]["stop_name"]
    finally:
        conn.close()


def _parse_carris_line_stop_query(user_message: str) -> Optional[Dict[str, Optional[str]]]:
    """Parses Carris wait/ETA questions for a specific line and stop."""
    query = user_message.strip()
    if not _query_has_wait_departure_intent(query):
        return None

    stop_id_match = re.search(r"\b(?:stop|paragem)\s+(?P<stop_id>\d{2,})\b", query, flags=re.IGNORECASE)
    line_match = re.search(r"\b(?P<line>\d{1,4}[A-Za-z]?)\b", query)

    eta_patterns = [
        r"\beta\s+(?:of\s+(?:route\s+)?)?(?P<line>\d{1,4}[A-Za-z]?)\s+(?:at|em|na|no)\s+(?P<stop>.+?)(?:[\?\!\.,;]|$)",
        r"quando chega\s+(?:o\s+)?(?P<line>\d{1,4}[A-Za-z]?)\s+(?:a|ao|à|na|no|em)\s+(?P<stop>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in eta_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {
                "kind": "eta",
                "line": match.group("line").upper(),
                "stop_name": _clean_query_fragment(match.group("stop")),
                "stop_id": stop_id_match.group("stop_id") if stop_id_match else None,
            }

    if re.search(r"(?:next\s+arrivals|arrivals|pr[oó]ximas\s+chegadas|chegadas)", query, flags=re.IGNORECASE) and stop_id_match:
        return {
            "kind": "arrivals",
            "line": None,
            "stop_name": None,
            "stop_id": stop_id_match.group("stop_id"),
        }

    stop_match = re.search(r"\b(?:at|em|na|no)\s+(?P<stop>[^\?\!\.,;]+)", query, flags=re.IGNORECASE)
    if line_match and stop_match:
        return {
            "kind": "departures",
            "line": line_match.group("line").upper(),
            "stop_name": _clean_query_fragment(stop_match.group("stop")),
            "stop_id": stop_id_match.group("stop_id") if stop_id_match else None,
        }

    return None


def _build_deterministic_metro_wait_response(
    user_message: str,
    context: str,
) -> Optional[str]:
    """Builds a deterministic single-station metro wait response without losing tool detail."""
    request = _parse_metro_wait_request(user_message)
    if not request:
        return None

    language = _infer_language(user_message, context)
    station = request.get("station") or ""
    direction = request.get("direction")
    status_requested = bool(request.get("status_requested"))

    from tools.metrolisboa_api import get_metro_wait_time, get_station_lines

    wait_args = {"station": station}
    if direction:
        wait_args["direction"] = direction

    try:
        wait_result = str(get_metro_wait_time.invoke(wait_args))
    except Exception:
        return None

    if wait_result.strip().startswith("❌"):
        return wait_result

    line_ids: List[str] = []
    if direction:
        shared_line = _get_line_id_between(station, direction)
        if shared_line:
            line_ids = [shared_line]
    if not line_ids:
        line_ids = list(dict.fromkeys(get_station_lines(station)))

    waits_section = "🗓️ **Próximos Metros** (tempo real):" if language == "pt" else "🗓️ **Next Metros** (real time):"
    source_label = "📌 **Fonte:**" if language == "pt" else "📌 **Source:**"
    updated_label = "**Atualizado:**" if language == "pt" else "**Updated:**"
    title = (
        f"🚇 **{station}** → **{direction}**"
        if direction
        else (
            f"🚇 **Próximo metro em {station}**"
            if language == "pt"
            else f"🚇 **Next metro at {station}**"
        )
    )

    response_lines = [title]
    if status_requested or line_ids:
        response_lines.append("⚠️ **Estado das Linhas:**" if language == "pt" else "⚠️ **Line Status:**")
        response_lines.extend(_build_route_state_lines(line_ids, language))
        response_lines.append("")

    response_lines.append(waits_section)
    if direction:
        response_lines.extend(_build_metro_wait_lines([(station, direction)], language))
    else:
        blocks = _parse_metro_wait_blocks(wait_result)
        station_label = "Estação" if language == "pt" else "Station"
        direction_label = "Direção" if language == "pt" else "Direction"
        next_label = "⏱️ Próximo Metro em:" if language == "pt" else "⏱️ Next Metro in:"
        if not blocks:
            response_lines.append("- Sem dados em tempo real" if language == "pt" else "- No real-time data available")
        for block in blocks:
            localized_times = [
                _localize_wait_times(time_text, language) for time_text in block.get("times", [])
            ]
            response_lines.append(
                f"- **{station_label} {station}:** {direction_label} {block['direction']} — **{next_label}** {' | '.join(localized_times)}"
            )
            note = _localize_platform_note(block.get("note"), language)
            if note:
                response_lines.append(f"  ℹ️ {note}")

    response_lines.append("")
    timestamp = extract_update_time(wait_result) or datetime.now().strftime("%H:%M")
    response_lines.append(
        f"{source_label} [*Metro de Lisboa*](https://www.metrolisboa.pt) | {updated_label} {timestamp}"
    )
    return "\n".join(response_lines).strip()


def _build_deterministic_carris_stop_response(user_message: str) -> Optional[str]:
    """Builds deterministic Carris stop/line wait responses from the best matching tool."""
    request = _parse_carris_line_stop_query(user_message)
    if not request:
        return None

    from tools.carris_api import (
        carris_get_arrivals,
        carris_get_next_departures,
        carris_vehicle_eta,
    )

    kind = request.get("kind")
    line = request.get("line")
    stop_id = request.get("stop_id")
    stop_name = request.get("stop_name")

    if not stop_id and stop_name:
        stop_id, resolved_stop_name = _resolve_carris_stop(stop_name)
        stop_name = resolved_stop_name or stop_name

    if kind == "arrivals" and stop_id:
        return str(carris_get_arrivals.invoke({"stop_id": stop_id, "limit": 8})).strip()

    if kind == "eta" and line and stop_name:
        return str(
            carris_vehicle_eta.invoke(
                {"route_short_name": line, "stop_name": stop_name}
            )
        ).strip()

    if kind == "departures" and line and stop_id:
        return str(
            carris_get_next_departures.invoke(
                {"stop_id": stop_id, "route_short_name": line, "limit": 8}
            )
        ).strip()

    return None


def _extract_cp_route_name(query: str) -> Optional[str]:
    """Extracts a canonical CP route name from a natural-language query."""
    normalized = _normalize_token(query)
    alias_map = {
        "linha de sintra": "Sintra",
        "sintra line": "Sintra",
        "sintra": "Sintra",
        "linha de cascais": "Cascais",
        "cascais line": "Cascais",
        "cascais": "Cascais",
        "linha da azambuja": "Azambuja",
        "azambuja line": "Azambuja",
        "azambuja": "Azambuja",
        "linha do sado": "Sado",
        "sado line": "Sado",
        "sado": "Sado",
    }

    for alias, canonical in alias_map.items():
        if alias in normalized:
            return canonical
    return None


def _build_cp_tool_spec(user_message: str) -> Optional[Dict[str, Any]]:
    """Maps common natural-language CP queries to deterministic tool specs."""
    query = user_message.strip()
    query_lower = query.lower()
    route_name = _extract_cp_route_name(query)
    has_train_context = bool(
        re.search(r"\b(cp|comboio|comboios|train|trains)\b", query_lower)
        or (
            route_name
            and re.search(
                r"\b(frequency|headway|how often|frequ[eê]ncia|de quanto em quanto tempo)\b",
                query_lower,
            )
        )
    )
    if not has_train_context:
        return None

    schedule_patterns = [
        r"(?:next|upcoming)\s+(?:train\s+)?departures\s+(?:from|at)\s+(?P<station>.+?)(?:[\?\!\.,;]|$)",
        r"(?:next|upcoming)\s+trains\s+(?:from|at)\s+(?P<station>.+?)(?:[\?\!\.,;]|$)",
        r"pr[oó]xim(?:os|as)\s+comboios\s+(?:de|em)\s+(?P<station>.+?)(?:[\?\!\.,;]|$)",
        r"hor[aá]rios?\s+dos?\s+comboios\s+(?:de|em)\s+(?P<station>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in schedule_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {
                "name": "get_train_schedule",
                "args": {"station_name": _resolve_cp_station_name(match.group("station"))},
            }

    if route_name and re.search(
        r"\b(frequency|headway|how often|frequ[eê]ncia|de quanto em quanto tempo)\b",
        query_lower,
    ):
        return {"name": "get_train_frequency", "args": {"route_name": route_name}}

    if re.search(r"\b(status|delay|delays|running|atrasos?|a funcionar)\b", query_lower):
        return {"name": "get_train_status", "args": {}}

    if re.search(
        r"(?:\b(cp|train|comboio)s?\b.*\b(routes|lines|linhas)\b)|(?:\b(routes|lines|linhas)\b.*\b(cp|train|comboio)s?\b)",
        query_lower,
    ):
        return {"name": "get_cp_routes", "args": {}}

    station_search_patterns = [
        r"(?:cp|train)\s+stations?\s+(?:for|near|named)\s+(?P<term>.+?)(?:[\?\!\.,;]|$)",
        r"esta[cç][õo]es?\s+cp\s+(?:para|de)\s+(?P<term>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in station_search_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {
                "name": "search_cp_stations",
                "args": {"query": _resolve_cp_station_name(match.group("term"))},
            }

    endpoints = _extract_route_endpoints(query)
    if endpoints and re.search(r"\b(cp|comboio|comboios|train|trains)\b", query_lower):
        return {
            "name": "plan_train_trip",
            "args": {
                "origin": _resolve_cp_station_name(endpoints[0]),
                "destination": _resolve_cp_station_name(endpoints[1]),
            },
        }

    return None


def _looks_like_carris_metropolitana_query(
    user_message: str,
    endpoints: Optional[Tuple[str, str]] = None,
) -> bool:
    """Heuristically identifies suburban-bus queries worth routing to Carris Metropolitana."""
    normalized = _normalize_token(user_message)
    if "carris metropolitana" in normalized:
        return True

    suburban_hints = {
        "almada",
        "cacilhas",
        "oeiras",
        "amadora",
        "sintra",
        "loures",
        "montijo",
        "seixal",
        "barreiro",
        "moita",
        "alcochete",
        "palmela",
        "sesimbra",
        "setubal",
        "almada forum",
    }

    if any(hint in normalized for hint in suburban_hints):
        return True

    if endpoints:
        endpoint_tokens = [_normalize_token(part) for part in endpoints]
        return any(hint in token for token in endpoint_tokens for hint in suburban_hints)

    return False


def _build_carris_metropolitana_tool_spec(user_message: str) -> Optional[Dict[str, Any]]:
    """Maps common Carris Metropolitana user requests to deterministic tool specs."""
    query = user_message.strip()
    query_lower = query.lower()
    endpoints = _extract_route_endpoints(query)
    has_cm_context = _looks_like_carris_metropolitana_query(query, endpoints=endpoints)
    if not has_cm_context:
        return None

    if re.search(r"\b(alerts?|alertas?|disruptions?)\b", query_lower):
        return {"name": "get_carris_metropolitana_alerts", "args": {}}

    stop_id_match = re.search(r"\b(?:stop id|stop|paragem)\s+(?P<stop_id>\d{3,})\b", query_lower)
    line_id_match = re.search(r"\b(?:line|linha)\s+(?P<line_id>\d{3,4}[a-z]?)\b", query_lower)

    if stop_id_match and re.search(r"\b(info|information|details|detalhes?)\b", query_lower):
        return {
            "name": "get_carris_metropolitana_stop_info",
            "args": {"stop_id": stop_id_match.group("stop_id")},
        }

    if stop_id_match and line_id_match and re.search(
        r"\b(next|departures|pr[oó]ximas?|partidas?)\b",
        query_lower,
    ):
        return {
            "name": "get_bus_next_departures",
            "args": {
                "line_id": line_id_match.group("line_id").upper(),
                "stop_id": stop_id_match.group("stop_id"),
            },
        }

    if line_id_match and re.search(
        r"\b(where|gps|location|locations|position|positions|onde|localiza[cç][aã]o|localiza[cç][õo]es)\b",
        query_lower,
    ):
        return {
            "name": "get_bus_realtime_locations",
            "args": {"line_id": line_id_match.group("line_id").upper()},
        }

    near_match = re.search(
        r"(?:near|perto de|around)\s+(?P<location>.+?)(?:[\?\!\.,;]|$)",
        query,
        flags=re.IGNORECASE,
    )
    if near_match and re.search(r"\b(bus|buses|autocarro|autocarros)\b", query_lower):
        return {
            "name": "get_real_time_bus_positions",
            "args": {"location": _clean_query_fragment(near_match.group("location")), "radius_km": 1.0},
        }

    if endpoints and re.search(r"\b(bus|buses|autocarro|autocarros)\b", query_lower):
        if re.search(r"\b(direct|diret[ao]s?)\b", query_lower):
            return {
                "name": "find_direct_bus_lines",
                "args": {"origin": endpoints[0], "destination": endpoints[1]},
            }
        return {
            "name": "find_bus_routes",
            "args": {"origin": endpoints[0], "destination": endpoints[1]},
        }

    line_search_patterns = [
        r"carris metropolitana\s+lines?\s+(?:for|to|near)\s+(?P<term>.+?)(?:[\?\!\.,;]|$)",
        r"linhas?\s+da\s+carris\s+metropolitana\s+(?:para|de)\s+(?P<term>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in line_search_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {
                "name": "search_carris_metropolitana_lines",
                "args": {"query": _clean_query_fragment(match.group("term"))},
            }

    return None


def _build_deterministic_metro_route_response(
    user_message: str,
    context: str,
) -> Optional[str]:
    """Builds a deterministic metro route answer directly from tool outputs."""
    endpoints = _extract_route_endpoints(user_message)
    if not endpoints:
        return None

    language = _infer_language(user_message, context)

    from tools.metrolisboa_api import METRO_LINES
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

    realtime_lines = _build_metro_wait_lines(
        _parse_wait_targets_from_route(route_result),
        language,
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

    if _normalize_token(endpoints[0]) == _normalize_token(endpoints[1]):
        return route_result

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
    for spec_builder in (
        _build_cp_tool_spec,
        _build_carris_metropolitana_tool_spec,
        _build_metro_tool_spec,
        _build_carris_urban_tool_spec,
    ):
        spec = spec_builder(query)
        if spec:
            return _build_tool_call(spec["name"], spec["args"])

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
        return _query_has_status_intent(user_message)

    def _run_direct_tool_fallback(self, user_message: str) -> Optional[str]:
        """Uses deterministic transport tools for broad status questions."""
        if not self._is_status_query(user_message) or _query_has_wait_departure_intent(user_message):
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

    def _resolve_deterministic_response(
        self,
        user_message: str,
        context: str = "",
        language: Optional[str] = None,
    ) -> Optional[str]:
        """Resolves deterministic direct-response fast paths shared by invoke() and build_subgraph()."""
        resolved_language = language or infer_response_language(user_query=user_message, default="en")

        deterministic_response = _build_deterministic_metro_route_response(
            user_message=user_message,
            context=context,
        )
        if deterministic_response:
            return finalize_worker_response(
                deterministic_response,
                agent_name="transport",
                user_query=user_message,
                language=resolved_language,
            )

        metro_wait_response = _build_deterministic_metro_wait_response(
            user_message=user_message,
            context=context,
        )
        if metro_wait_response:
            return finalize_worker_response(
                metro_wait_response,
                agent_name="transport",
                user_query=user_message,
                language=resolved_language,
            )

        carris_stop_response = _build_deterministic_carris_stop_response(user_message)
        if carris_stop_response:
            return finalize_worker_response(
                carris_stop_response,
                agent_name="transport",
                user_query=user_message,
                language=resolved_language,
            )

        cp_tool_spec = _build_cp_tool_spec(user_message)
        carris_metropolitana_tool_spec = _build_carris_metropolitana_tool_spec(user_message)

        if not cp_tool_spec and not carris_metropolitana_tool_spec:
            direct_route_response = _build_deterministic_route_tool_response(user_message)
            if direct_route_response:
                return finalize_worker_response(
                    direct_route_response,
                    agent_name="transport",
                    user_query=user_message,
                    language=resolved_language,
                )

        direct_tool_response = self._run_direct_tool_fallback(user_message)
        if direct_tool_response:
            return finalize_worker_response(
                direct_tool_response,
                agent_name="transport",
                user_query=user_message,
                language=resolved_language,
            )

        return None

    def _invoke_deterministic_tool_call(
        self,
        user_message: str,
        language: Optional[str] = None,
    ) -> Optional[str]:
        """Invokes a deterministic single-tool fast path and finalizes the result for the user."""
        tool_call_msg = _build_deterministic_transport_tool_call(user_message)
        if not tool_call_msg or not tool_call_msg.tool_calls:
            return None

        tool_call = tool_call_msg.tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool = self._get_tool_by_name(tool_name)
        if not tool:
            return None

        result = tool.invoke(tool_args)
        resolved_language = language or infer_response_language(user_query=user_message, default="en")
        return finalize_worker_response(
            str(result).strip(),
            agent_name="transport",
            user_query=user_message,
            language=resolved_language,
        )

    def _build_subgraph_deterministic_tool_call(
        self,
        user_message: str,
    ) -> Optional[AIMessage]:
        """Builds deterministic tool-call messages for subgraph execution and coverage tracking."""
        tool_call_msg = _build_deterministic_transport_tool_call(user_message)
        if tool_call_msg:
            return tool_call_msg

        query_lower = user_message.lower()
        if (
            not _query_has_wait_departure_intent(user_message)
            and re.search(r"\b(summary|overview|all transport|transport summary|transport overview|across)\b", query_lower)
            and re.search(r"\b(transport|metro|bus|buses|train|trains|comboio|comboios|autocarro|autocarros)\b", query_lower)
        ):
            return _build_tool_call("get_transport_summary", {})

        endpoints = _extract_route_endpoints(user_message)
        if endpoints:
            return _build_tool_call(
                "get_route_between_stations",
                {"origin": endpoints[0], "destination": endpoints[1]},
            )

        return None

    @staticmethod
    def _is_deterministic_subgraph_tool_result(messages: List) -> bool:
        """Returns True when the current ToolMessage came from an auto-generated fast-path tool call."""
        if len(messages) < 2 or not isinstance(messages[-1], ToolMessage):
            return False

        previous_message = messages[-2]
        if not isinstance(previous_message, AIMessage) or not getattr(previous_message, "tool_calls", None):
            return False

        tool_call_id = previous_message.tool_calls[0].get("id", "")
        return isinstance(tool_call_id, str) and tool_call_id.startswith("auto_")

    @staticmethod
    def _build_language_instruction(language: str) -> str:
        """Builds a compact language instruction for LLM-backed subgraph steps."""
        return (
            "Respond ENTIRELY in Portuguese (PT-PT)."
            if language == "pt"
            else "Respond ENTIRELY in English."
        )

    def _ensure_subgraph_messages(
        self,
        messages: List,
        language: str,
    ) -> List:
        """Ensures transport subgraph LLM steps receive system and language instructions."""
        updated_messages = list(messages)
        if not updated_messages or not isinstance(updated_messages[0], SystemMessage):
            updated_messages = [SystemMessage(content=self.system_prompt)] + updated_messages

        if not any(
            isinstance(message, SystemMessage)
            and "Respond ENTIRELY" in str(message.content)
            for message in updated_messages[:3]
        ):
            updated_messages = [
                updated_messages[0],
                SystemMessage(content=self._build_language_instruction(language)),
                *updated_messages[1:],
            ]

        return updated_messages

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

        language = infer_response_language(user_query=user_message, default="en")
        realtime_lines = _build_metro_wait_lines(targets, language=language)

        return _upsert_realtime_wait_section(response, realtime_lines, language=language)

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
            deterministic_response = self._resolve_deterministic_response(
                user_message=user_message,
                context=context,
                language=language,
            )
            if deterministic_response:
                return deterministic_response

            deterministic_tool_response = self._invoke_deterministic_tool_call(
                user_message=user_message,
                language=language,
            )
            if deterministic_tool_response:
                return deterministic_tool_response

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

            user_message = None
            for message in reversed(messages):
                if isinstance(message, HumanMessage) and message.content:
                    user_message = str(message.content)
                    break

            language = infer_response_language(user_query=user_message or "", default="en")

            last_message = messages[-1] if messages else None
            if isinstance(last_message, ToolMessage):
                if self._is_deterministic_subgraph_tool_result(messages):
                    finalized_response = finalize_worker_response(
                        str(last_message.content).strip(),
                        agent_name="transport",
                        user_query=user_message or "",
                        language=language,
                    )
                    return {"messages": [AIMessage(content=finalized_response)]}

                response = self._safe_llm_invoke(
                    self.llm_with_tools,
                    self._ensure_subgraph_messages(messages, language),
                )
                return {"messages": [response]}

            if user_message:
                deterministic_tool_call = self._build_subgraph_deterministic_tool_call(user_message)
                if deterministic_tool_call:
                    return {"messages": [deterministic_tool_call]}

                deterministic_response = self._resolve_deterministic_response(
                    user_message=user_message,
                    context="",
                    language=language,
                )
                if deterministic_response:
                    return {"messages": [AIMessage(content=deterministic_response)]}

            response = self._safe_llm_invoke(
                self.llm_with_tools,
                self._ensure_subgraph_messages(messages, language),
            )
            return {"messages": [response]}

        def should_continue(state: AgentState) -> str:
            """Determines next step."""
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)

        def fallback_tools_node(state: AgentState) -> dict:
            """Executes tool calls without LangGraph's ToolNode when tests inject mock-like tools."""
            last_message = state["messages"][-1]
            tool_messages: List[ToolMessage] = []

            for tool_call in getattr(last_message, "tool_calls", []) or []:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", f"auto_{uuid.uuid4().hex}")
                tool = self._get_tool_by_name(tool_name)

                if tool is None:
                    result = f"Tool '{tool_name}' not found."
                else:
                    try:
                        result = tool.invoke(tool_args)
                    except Exception as exc:
                        result = f"Error executing {tool_name}: {exc}"

                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )

            return {"messages": tool_messages}

        try:
            tools_node = ToolNode(self.tools)
        except ValueError:
            tools_node = fallback_tools_node

        workflow.add_node("tools", tools_node)
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

        test_queries = [
            "Is the metro working?",
            "When is the next metro at Saldanha towards Odivelas?",
            "What are the next departures for 732 at Rossio?",
            "When are the next trains from Entrecampos?",
            "How do I get from Rossio to Sintra by train?",
            "Show real-time Carris Metropolitana buses near Almada",
            "What are the direct Carris Metropolitana buses from Oeiras to Amadora?",
        ]

        for query in test_queries:
            print(f"\n\033[1m📝 Testing query:\033[0m '{query}'")
            deterministic_call = _build_deterministic_transport_tool_call(query)
            if deterministic_call and deterministic_call.tool_calls:
                call = deterministic_call.tool_calls[0]
                print(
                    f"   🔧 Deterministic tool call: {call.get('name')}({call.get('args', {})})"
                )
            else:
                print("   🔧 Deterministic tool call: none (direct response or LLM path)")

            response = agent.invoke(query)
            print("\n\033[1m🤖 Response:\033[0m")
            print(response[:1500])

        print("\n\033[1;32m✅ Transport agent working!\033[0m")

    except Exception as e:
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
        import traceback

        traceback.print_exc()
