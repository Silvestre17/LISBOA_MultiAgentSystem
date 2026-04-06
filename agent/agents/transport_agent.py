# ==========================================================================
# Master Thesis - Transport Agent
#   - André Filipe Gomes Silvestre, 20240502
#
#   Specialized agent for transport-related queries.
#   Handles metro, bus, train, and multi-modal routing.
#   Uses BaseAgent.execute_react_loop() for tool execution.
# ==========================================================================

import re
import unicodedata
import uuid
from copy import deepcopy
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

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
        r"\b(agora|pfv|por favor|sff|please|now|right now|já|ja|mesmo|pff|only|just|apenas|s[oó])\b.*$",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"\b(without(?:\s+taking)?\s+(?:a\s+)?(?:bus|buses|tram|trams)|sem\s+(?:autocarro|autocarros|eletrico|el[eé]trico|eletricos|el[eé]tricos)|(?:bus|tram)\s+only|s[oó]\s+de\s+(?:autocarro|autocarros|eletrico|el[eé]trico|eletricos|el[eé]tricos))\b.*$",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"\b(?:sem\s+complica(?:[çc](?:ão|ao|ões|oes))|qual(?:\s+[ée])?\s+a?\s*melhor\s+(?:forma|rota|caminho)|what(?:'s| is)\s+the\s+best\s+(?:way|route)|best\s+(?:way|route)|o\s+metro\s+serve\s+bem|serve\s+bem)\b.*$",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"\b(?:de metro|de autocarro|de comboio|de elétrico|de eletrico|(?:by|using|via)(?:\s+the)?\s+(?:metro|bus|tram|train))\b",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"^(?:metro|bus|tram|train|comboio|autocarro|el[eé]trico)\s+de\s+",
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
    part = re.sub(r"\b(at[eé]|ate)\s*$", "", part, flags=re.IGNORECASE)
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
    from tools.cp_api import CP_KEY_STATIONS, load_cp_aml_stations

    alias_map: Dict[str, str] = {}
    for key, info in CP_KEY_STATIONS.items():
        canonical = str(info.get("name") or key.replace("_", " ").title()).strip()
        alias_map[_normalize_token(canonical)] = canonical
        alias_map[_normalize_token(key.replace("_", " "))] = canonical

    for station in load_cp_aml_stations().values():
        canonical = str(station.get("name") or "").strip()
        if canonical:
            alias_map[_normalize_token(canonical)] = canonical

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
    official_api_temporarily_unavailable = False
    for station, direction in targets[:2]:
        try:
            wait_result = str(
                get_metro_wait_time.invoke(
                    {"station": station, "direction": direction}
                )
            )
        except Exception:
            wait_result = ""

        wait_result_normalized = _normalize_token(wait_result)
        if wait_result.strip().startswith("❌") and any(
            marker in wait_result_normalized
            for marker in [
                "temporarily unavailable",
                "temporariamente indisponivel",
                "not responding right now",
                "official metro",
                "oficial metro",
                "fallback endpoint still provides line status",
            ]
        ):
            official_api_temporarily_unavailable = True
            continue

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
        if official_api_temporarily_unavailable:
            if language == "pt":
                realtime_lines.extend(
                    [
                        "- Dados oficiais do Metro em tempo real estão temporariamente indisponíveis.",
                        "  ℹ️ O estado das linhas continua disponível, mas os tempos de espera não puderam ser confirmados agora.",
                    ]
                )
            else:
                realtime_lines.extend(
                    [
                        "- Official Metro real-time data is temporarily unavailable.",
                        "  ℹ️ Line status is still available, but live wait times could not be confirmed right now.",
                    ]
                )
        else:
            realtime_lines.append(
                "- Sem dados em tempo real" if language == "pt" else "- No real-time data available"
            )

    return realtime_lines


def _extract_route_endpoints(user_message: str) -> Optional[Tuple[str, str]]:
    """Extracts route endpoints from common PT/EN route phrasings."""
    patterns = [
        r"\b(?:plan|planeia|planeie|organiza|organize)\b.*?\b(?:em|in)\s+(?P<destination>[^,\?\!\.]+),\s*(?:diz[- ]me|diga[- ]me|tell me|show me)\s+como\s+l[aá]\s+chegar\s+a\s+partir\s+d(?:o|a)\s+(?P<origin>.+?)(?:\s+e\b|[\?\!\.,;]|$)",
        r"\b(?:plan|planeia|planeie|organiza|organize)\b.*?\b(?:em|in)\s+(?P<destination>[^,\?\!\.]+),\s*(?:tell me|show me)\s+how\s+to\s+get\s+there\s+from\s+(?P<origin>.+?)(?:\s+and\b|[\?\!\.,;]|$)",
        r"\b(?:quero|preciso|tenho)\s+(?:de\s+)?ir\s+(?:para|ao|a|à)\s+(?P<destination>.+?)\s+a\s+partir\s+d(?:o|a)\s+(?P<origin>.+?)(?:[\?\!\.,;]|$)",
        r"\b(?:a\s+partir\s+d(?:o|a)|desde)\s+(?P<origin>.+?)\s+(?:para|ao|a|à)\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\b(?:tou|estou)\s+n(?:o|a)\s+(?P<origin>.+?)\s+(?:e\s+)?(?:preciso|quero|tenho)\s+(?:de\s+)?ir\s+(?:para|ao|a|à)\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\b(?:need|want)\s+to\s+go\s+to\s+(?P<destination>.+?)\s+from\s+(?P<origin>.+?)(?:[\?\!\.,;]|$)",
        r"\b(?:i'?m|i am)\s+(?:at|in)\s+(?P<origin>.+?)\s+and\s+(?:i\s+)?(?:need|want)\s+to\s+go\s+to\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bde\s+metro\s+de\s+(?P<origin>.+?)\s+at[eé]\s+(?:a|ao|à)?\s*(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bde\s+metro\s+de\s+(?P<origin>.+?)\s+para\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bde\s+(?P<origin>.+?)\s+at[eé]\s+(?:a|ao|à)?\s*(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bde\s+(?P<origin>.+?)\s+para\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bdo\s+(?P<origin>.+?)\s+ao\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bdo\s+(?P<origin>.+?)\s+para\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bda\s+(?P<origin>.+?)\s+à\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bda\s+(?P<origin>.+?)\s+para\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
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


def _parse_route_mode_preferences(user_message: str) -> Dict[str, bool]:
    """Parses explicit route-mode constraints such as bus-only or without-bus requests."""
    normalized = _normalize_token(user_message)

    bus_only_phrases = [
        "bus only",
        "only bus",
        "only buses",
        "by bus only",
        "only by bus",
        "so autocarro",
        "so autocarros",
        "só autocarro",
        "só autocarros",
        "so de autocarro",
        "só de autocarro",
        "so de autocarros",
        "só de autocarros",
        "apenas autocarro",
        "apenas autocarros",
        "apenas de autocarro",
        "apenas de autocarros",
    ]
    tram_only_phrases = [
        "tram only",
        "only tram",
        "only trams",
        "by tram only",
        "only by tram",
        "so eletrico",
        "so eletricos",
        "so electrico",
        "só eletrico",
        "só elétrico",
        "só eletricos",
        "só elétricos",
        "so de eletrico",
        "só de elétrico",
        "so de eletricos",
        "só de elétricos",
        "apenas eletrico",
        "apenas elétrico",
        "apenas eletricos",
        "apenas elétricos",
        "apenas de eletrico",
        "apenas de elétrico",
    ]
    metro_only_phrases = [
        "metro only",
        "only metro",
        "only by metro",
        "by metro only",
        "so metro",
        "só metro",
        "so de metro",
        "só de metro",
        "apenas metro",
        "apenas de metro",
    ]
    exclude_bus_phrases = [
        "without bus",
        "without buses",
        "without taking a bus",
        "without taking buses",
        "without using a bus",
        "without using buses",
        "no bus",
        "no buses",
        "not by bus",
        "dont want bus",
        "do not want bus",
        "sem autocarro",
        "sem autocarros",
        "nao quero autocarro",
        "não quero autocarro",
        "nao quero autocarros",
        "não quero autocarros",
    ]
    exclude_tram_phrases = [
        "without tram",
        "without trams",
        "without taking a tram",
        "without taking trams",
        "no tram",
        "no trams",
        "not by tram",
        "sem eletrico",
        "sem elétrico",
        "sem eletricos",
        "sem elétricos",
        "sem electrico",
        "nao quero eletrico",
        "não quero elétrico",
        "nao quero eletricos",
        "não quero elétricos",
    ]
    exclude_metro_phrases = [
        "without metro",
        "without taking the metro",
        "without taking metro",
        "without using the metro",
        "no metro",
        "not by metro",
        "avoid metro",
        "sem metro",
        "nao quero metro",
        "não quero metro",
        "nao usar metro",
        "não usar metro",
    ]

    bus_only = any(phrase in normalized for phrase in bus_only_phrases)
    tram_only = any(phrase in normalized for phrase in tram_only_phrases)
    metro_only = any(phrase in normalized for phrase in metro_only_phrases)
    exclude_bus = any(phrase in normalized for phrase in exclude_bus_phrases)
    exclude_tram = any(phrase in normalized for phrase in exclude_tram_phrases)
    exclude_metro = any(phrase in normalized for phrase in exclude_metro_phrases)

    bus_only = bus_only or bool(
        re.search(
            r"\b(?:only|just|apenas|s[oó])\b(?:\s+de)?\s+(?:bus|buses|autocarro|autocarros)\b",
            normalized,
        )
    )
    tram_only = tram_only or bool(
        re.search(
            r"\b(?:only|just|apenas|s[oó])\b(?:\s+de)?\s+(?:tram|trams|eletrico|eletricos|electrico|electricos)\b",
            normalized,
        )
    )
    metro_only = metro_only or bool(
        re.search(
            r"\b(?:only|just|apenas|s[oó])\b(?:\s+de)?\s+metro\b",
            normalized,
        )
    )

    if exclude_tram and re.search(r"\b(bus|buses|autocarro|autocarros)\b", normalized):
        bus_only = True
    if exclude_bus and re.search(r"\b(tram|trams|eletrico|eletricos|electrico|electricos)\b", normalized):
        tram_only = True
    if exclude_metro and re.search(r"\b(bus|buses|autocarro|autocarros)\b", normalized):
        bus_only = True
    if exclude_metro and re.search(r"\b(tram|trams|eletrico|eletricos|electrico|electricos)\b", normalized):
        tram_only = True

    return {
        "bus_only": bus_only,
        "tram_only": tram_only,
        "metro_only": metro_only,
        "exclude_bus": exclude_bus,
        "exclude_tram": exclude_tram,
        "exclude_metro": exclude_metro,
    }


def _query_has_route_mode_constraints(user_message: str) -> bool:
    """Returns whether the user explicitly constrained the allowed transport modes for a route."""
    preferences = _parse_route_mode_preferences(user_message)
    return any(preferences.values())


def _tool_result_indicates_no_match(result: str) -> bool:
    """Detects tool outputs that mean no valid route or line was found."""
    normalized = _normalize_token(result or "")
    negative_markers = [
        "no direct carris route found",
        "could not locate",
        "sem linhas diretas",
        "sem linha direta",
        "no direct bus routes found",
        "no direct buses",
        "nao consegui",
        "não consegui",
        "not found",
        "no bus stops found",
    ]
    return any(marker in normalized for marker in negative_markers)


def _extract_carris_mode_section(route_result: str, section_name: str) -> str:
    """Extracts the requested BUSES or TRAMS section from Carris Urban route output."""
    lines = (route_result or "").splitlines()
    target = section_name.upper().strip()
    collecting = False
    section_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped in {"TRAMS", "BUSES"}:
            if collecting and stripped != target:
                break
            collecting = stripped == target
            if collecting:
                section_lines.append(line)
            continue

        if collecting:
            section_lines.append(line)

    while section_lines and not section_lines[-1].strip():
        section_lines.pop()

    return "\n".join(section_lines).strip()


def _carris_section_has_routes(section: str) -> bool:
    """Returns whether a Carris section contains at least one concrete route line."""
    return bool(re.search(r"^\s*\d{1,4}[A-Z]?\s*:", section or "", flags=re.MULTILINE))


def _parse_carris_route_entries(section: str) -> List[Dict[str, Any]]:
    """Parses a Carris route section into route entries with summary and detail lines."""
    entries: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for raw_line in (section or "").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped in {"BUSES", "TRAMS"} or re.fullmatch(r"[-=]{3,}", stripped):
            continue

        route_match = re.match(r"^(\d{1,4}[A-Z]?)\s*:\s*(.+)$", stripped)
        if route_match:
            if current:
                entries.append(current)
            current = {
                "route": route_match.group(1),
                "summary": route_match.group(2).strip(),
                "details": [],
            }
            continue

        if current:
            current["details"].append(stripped)

    if current:
        entries.append(current)

    return entries


def _localize_period_label(label: str, language: str) -> str:
    """Localizes Carris service-period labels."""
    mapping = {
        "morning": ("manhã", "morning"),
        "midday": ("meio do dia", "midday"),
        "afternoon": ("tarde", "afternoon"),
        "evening": ("fim do dia", "evening"),
        "night": ("noite", "night"),
    }
    normalized = _normalize_token(label)
    if normalized in mapping:
        return mapping[normalized][0 if language == "pt" else 1]
    return label.strip()


def _minutes_from_clock(clock_text: str) -> Optional[int]:
    """Converts HH:MM clock text into minutes since midnight."""
    match = re.match(r"^(\d{2}):(\d{2})$", clock_text.strip())
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def _summarize_carris_frequency_result(
    frequency_result: str,
    language: str,
) -> List[str]:
    """Builds concise user-facing bullets from the Carris frequency tool output."""
    if not frequency_result:
        return []

    no_service_match = re.search(
        r"No scheduled trips found for route '?(?P<route>[^']+)'? today",
        frequency_result,
        flags=re.IGNORECASE,
    )
    if no_service_match:
        if language == "pt":
            return [
                "  - **Serviço programado hoje:** não foram encontradas partidas na base GTFS.",
                "  - **Como confirmar:** peça-me as próximas partidas numa paragem específica desta linha ou confirme em carris.pt.",
            ]
        return [
            "  - **Scheduled service today:** no departures were found in the GTFS timetable.",
            "  - **How to confirm:** ask me for departures at a specific stop on this line or check carris.pt.",
        ]

    total_match = re.search(r"Total de passagens hoje:\s*(\d+)", frequency_result, flags=re.IGNORECASE)
    lines = frequency_result.splitlines()
    now_minutes = datetime.now().hour * 60 + datetime.now().minute
    selected_summary: Optional[str] = None

    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        period_match = re.match(
            r"^[🌅☀️🌤️🌙🌃]\s*([A-Za-zÀ-ÿ ]+)\s*\((\d{2}:\d{2})-(\d{2}:\d{2})\)(?::\s*(.+))?$",
            stripped,
        )
        if not period_match:
            index += 1
            continue

        label = _localize_period_label(period_match.group(1), language)
        start_clock = period_match.group(2)
        end_clock = period_match.group(3)
        inline_summary = (period_match.group(4) or "").strip()
        start_minutes = _minutes_from_clock(start_clock)
        end_minutes = _minutes_from_clock(end_clock)
        is_current_period = (
            start_minutes is not None
            and end_minutes is not None
            and start_minutes <= now_minutes <= end_minutes
        )

        candidate_summary: Optional[str] = None
        if inline_summary and "sem serviço" not in _normalize_token(inline_summary):
            if language == "pt":
                candidate_summary = f"**Serviço programado:** {label}, {inline_summary}."
            else:
                candidate_summary = f"**Scheduled service:** {label}, {inline_summary}."
        else:
            window_lines: List[str] = []
            look_ahead = index + 1
            while look_ahead < len(lines):
                next_line = lines[look_ahead].strip()
                if re.match(r"^[🌅☀️🌤️🌙🌃]", next_line):
                    break
                if next_line:
                    window_lines.append(next_line)
                look_ahead += 1

            avg_match = next(
                (
                    re.search(r"Frequência média:\s*\*\*(.+?)\*\*", line, flags=re.IGNORECASE)
                    for line in window_lines
                    if "Frequência média" in line
                ),
                None,
            )
            first_last_match = next(
                (
                    re.search(r"Primeiro:\s*([^|]+)\|\s*Último:\s*(.+)$", line, flags=re.IGNORECASE)
                    for line in window_lines
                    if "Primeiro:" in line and "Último:" in line
                ),
                None,
            )

            if avg_match:
                avg_value = avg_match.group(1).strip()
                first_last_text = ""
                if first_last_match:
                    first_trip = first_last_match.group(1).strip()
                    last_trip = first_last_match.group(2).strip()
                    if language == "pt":
                        first_last_text = f" entre {first_trip} e {last_trip}"
                    else:
                        first_last_text = f" between {first_trip} and {last_trip}"
                if language == "pt":
                    candidate_summary = f"**Frequência programada:** {label}, média de {avg_value}{first_last_text}."
                else:
                    candidate_summary = f"**Scheduled frequency:** {label}, about {avg_value}{first_last_text}."

            index = look_ahead - 1

        if not candidate_summary:
            index += 1
            continue

        if is_current_period:
            selected_summary = candidate_summary
            break
        if selected_summary is None:
            selected_summary = candidate_summary

        index += 1

    summary_lines: List[str] = []
    if total_match:
        total = total_match.group(1)
        if language == "pt":
            summary_lines.append(f"  - **Passagens programadas hoje:** {total}")
        else:
            summary_lines.append(f"  - **Scheduled departures today:** {total}")
    if selected_summary:
        summary_lines.append(f"  - {selected_summary}")

    return summary_lines


def _format_carris_route_detail(
    detail: str,
    language: str,
    route: Optional[str] = None,
    frequency_lookup: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """Formats a single Carris route detail line into nested markdown bullets."""
    stripped = (detail or "").strip()
    if not stripped:
        return []

    if re.fullmatch(r"check schedule\.?", stripped, re.IGNORECASE):
        frequency_result = frequency_lookup(route) if frequency_lookup and route else ""
        frequency_summary = _summarize_carris_frequency_result(frequency_result, language)
        if frequency_summary:
            return frequency_summary
        if language == "pt":
            return [
                "  - **Horário detalhado:** esta rota não devolveu uma paragem específica para calcular partidas exatas.",
                "  - **Como confirmar:** peça-me horários numa paragem concreta desta linha.",
            ]
        return [
            "  - **Detailed timetable:** this route result did not include a specific stop for exact departures.",
            "  - **How to confirm:** ask me for departures at a specific stop on this line.",
        ]

    next_match = re.match(r"^(?:\*\*)?Next(?:\*\*)?:\s*(.+)$", stripped, re.IGNORECASE)
    if next_match:
        detail_text = next_match.group(1).strip()
        stop_match = re.search(r"\(stop\s+([^)]+)\)", detail_text, re.IGNORECASE)
        stop_name = stop_match.group(1).strip() if stop_match else ""
        times_text = re.sub(r"\(stop\s+([^)]+)\)", "", detail_text, flags=re.IGNORECASE).strip()
        lines: List[str] = []
        if times_text:
            if language == "pt":
                lines.append(f"  - **Próximas partidas:** {times_text}")
            else:
                lines.append(f"  - **Next departures:** {times_text}")
        if stop_name:
            if language == "pt":
                lines.append(f"  - **Paragem:** {stop_name}")
            else:
                lines.append(f"  - **Stop:** {stop_name}")
        return lines

    travel_match = re.match(r"^~\s*(\d+)\s*min(?:\s*travel)?$", stripped, re.IGNORECASE)
    if travel_match:
        travel_text = f"~{travel_match.group(1)} min"
        if language == "pt":
            return [f"  - **Tempo estimado:** {travel_text}"]
        return [f"  - **Estimated travel time:** {travel_text}"]

    return [f"  - {stripped}"]


def _format_carris_mode_section_markdown(
    section: str,
    language: str,
    frequency_lookup: Optional[Callable[[str], str]] = None,
) -> str:
    """Formats a Carris BUSES/TRAMS section into clean markdown bullets."""
    entries = _parse_carris_route_entries(section)
    if not entries:
        return ""

    blocks: List[str] = []
    for entry in entries:
        block_lines = [f"- **{entry['route']}**: {entry['summary']}"]
        for detail in entry.get("details", []):
            block_lines.extend(
                _format_carris_route_detail(
                    detail,
                    language,
                    route=entry.get("route"),
                    frequency_lookup=frequency_lookup,
                )
            )
        blocks.append("\n".join(block_lines))

    return "\n\n".join(blocks).strip()


def _clean_metropolitana_direct_bus_block(text: str) -> str:
    """Removes wrapper lines from Carris Metropolitana direct-bus output for embedding in summaries."""
    cleaned_lines: List[str] = []
    for raw_line in (text or "").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"[-=]{3,}", stripped):
            continue
        if stripped.startswith("🚌 **Autocarros:") or stripped.startswith("🚌 **Buses:"):
            continue
        if "linha(s) direta(s) encontrada(s):" in stripped or "direct line(s) found:" in stripped:
            continue
        if stripped.startswith("💡 **Como usar:") or stripped.startswith("💡 **How to use"):
            break
        if stripped.startswith("📌 **Fonte:") or stripped.startswith("📌 **Source:"):
            continue
        if stripped.startswith("⚠️ Scope:") or stripped.startswith("💡 For Lisbon city-only"):
            continue
        cleaned_lines.append(raw_line)

    return "\n".join(cleaned_lines).strip()


def _parse_wait_targets_from_route(route_result: str) -> List[Tuple[str, str]]:
    """Parses the stations and directions needed for live metro wait times."""
    details = _parse_route_details(route_result)
    directions = details.get("directions", [])
    targets: List[Tuple[str, str]] = []

    if details.get("board_station") and directions:
        targets.append((details["board_station"], directions[0]))

    if details.get("transfer_station") and len(directions) >= 2:
        targets.append((details["transfer_station"], directions[1]))

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


def _is_future_transport_planning_query(user_message: str) -> bool:
    """Returns whether the transport query is clearly about future planning.

    Args:
        user_message: Original transport query.

    Returns:
        bool: True when the query targets a future day or planning horizon and
        should therefore avoid live wait/departure times.
    """
    normalized = _normalize_token(user_message)
    if not normalized:
        return False

    for pattern, date_format in (
        (r"\b(202\d-\d{2}-\d{2})\b", "%Y-%m-%d"),
        (r"\b(\d{2}/\d{2}/202\d)\b", "%d/%m/%Y"),
    ):
        match = re.search(pattern, user_message)
        if not match:
            continue
        try:
            explicit_date = datetime.strptime(match.group(1), date_format).date()
        except ValueError:
            continue
        if explicit_date > datetime.now().date():
            return True

    future_patterns = [
        r"\btomorrow\b",
        r"\bamanha\b",
        r"\bthis weekend\b",
        r"\bweekend\b",
        r"\bfim de semana\b",
        r"\bnext week\b",
        r"\bproxima semana\b",
        r"\bnext month\b",
        r"\bproximo mes\b",
        r"\b(?:2|3|4|5)\s+days?\b",
        r"\b(?:2|3|4|5)\s+dias?\b",
        r"\bday\s+[23]\b",
        r"\b[23](?:o|a)?\s+dia\b",
        r"\bplan\b",
        r"\bplanning\b",
        r"\broteiro\b",
        r"\bitinerario\b",
        r"\bitinerary\b",
    ]
    return any(re.search(pattern, normalized) for pattern in future_patterns)


def _build_future_realtime_note(language: str) -> str:
    """Builds a localized note explaining why live waits were omitted.

    Args:
        language: Preferred response language.

    Returns:
        str: Localized future-planning note.
    """
    if language == "pt":
        return (
            "- 🗓️ **Planeamento futuro:** omiti tempos em tempo real porque esta pergunta é para outro momento. "
            "Confirme esperas e partidas no próprio dia em metrolisboa.pt."
        )
    return (
        "- 🗓️ **Future planning:** I omitted real-time waits because this request is for a later trip. "
        "Check live waits and departures on the day itself at metrolisboa.pt."
    )


def _build_future_metro_wait_limit_response(
    station: str,
    direction: Optional[str],
    language: str,
) -> str:
    """Builds a localized response for future metro wait-time queries.

    Args:
        station: Requested station.
        direction: Optional direction constraint.
        language: Preferred response language.

    Returns:
        str: Localized response explaining that live wait times are not shown for
        future planning.
    """
    title = (
        f"🚇 **{station}** → **{direction}**"
        if direction
        else (
            f"🚇 **Próximo metro em {station}**"
            if language == "pt"
            else f"🚇 **Next metro at {station}**"
        )
    )
    source_label = "📌 **Fonte:**" if language == "pt" else "📌 **Source:**"
    updated_label = "**Atualizado:**" if language == "pt" else "**Updated:**"

    if language == "pt":
        note = (
            "- ⚠️ **Tempo real indisponível para planeamento futuro:** os tempos de espera do Metro só são úteis no próprio momento da viagem."
        )
    else:
        note = (
            "- ⚠️ **Real-time waits are not shown for future planning:** Metro wait times are only meaningful close to the trip itself."
        )

    return "\n".join(
        [
            title,
            "",
            note,
            _build_future_realtime_note(language),
            "",
            f"{source_label} [*Metro de Lisboa*](https://www.metrolisboa.pt) | {updated_label} {datetime.now().strftime('%H:%M')}",
        ]
    ).strip()


def _join_transport_labels(labels: List[str], language: str) -> str:
    """Join transport-scope labels into a user-facing list."""
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    separator = " e " if language == "pt" else " and "
    return ", ".join(labels[:-1]) + separator + labels[-1]


def _detect_unsupported_transport_modes(user_message: str) -> List[str]:
    """Detect transport modes that the current runtime does not verify directly."""
    normalized = _normalize_token(user_message)
    if not normalized:
        return []

    supported_network_hint = bool(
        re.search(
            r"\b(metro|carris|autocarro|autocarros|bus|buses|comboio|comboios|train|trains|cp|tram|trams|eletrico|eletricos|electrico|electricos)\b",
            normalized,
        )
    )
    ride_hailing_detail_hint = bool(
        re.search(
            r"\b(price|cost|fare|estimate|estimated|eta|wait time|waiting time|how much|quanto custa|preco|preço|tarifa|espera|tempo de espera)\b",
            normalized,
        )
    )

    unsupported_modes: List[str] = []

    def _append(mode_code: str) -> None:
        if mode_code not in unsupported_modes:
            unsupported_modes.append(mode_code)

    if re.search(r"\b(ferry|ferries|boat|boats|barco|barcos|transtejo|soflusa|cacilheiro|cacilheiros)\b", normalized):
        _append("ferries")

    if re.search(r"\bfertagus\b", normalized):
        _append("fertagus")

    if (
        re.search(
            r"\b(gira|bike|bikes|bicycle|bicycles|bicicleta|bicicletas|scooter|scooters|e-scooter|e-scooters|e scooter|e scooters|trotinete|trotinetes)\b",
            normalized,
        )
        and not supported_network_hint
    ):
        _append("micromobility")

    if re.search(r"\b(uber|bolt|taxi|taxis|táxi|táxis)\b", normalized) and (
        ride_hailing_detail_hint or not supported_network_hint
    ):
        _append("ride_hailing")

    return unsupported_modes


def _build_unsupported_transport_scope_response(
    user_message: str,
    language: str,
) -> Optional[str]:
    """Build an honest limitation note for unsupported transport networks or modes."""
    unsupported_modes = _detect_unsupported_transport_modes(user_message)
    if not unsupported_modes:
        return None

    labels_en = {
        "ferries": "Transtejo/Soflusa ferries",
        "fertagus": "Fertagus trains",
        "micromobility": "Gira bikes or shared e-scooters",
        "ride_hailing": "ride-hailing or taxi pricing details",
    }
    labels_pt = {
        "ferries": "ferries Transtejo/Soflusa",
        "fertagus": "comboios Fertagus",
        "micromobility": "bicicletas Gira ou trotinetes partilhadas",
        "ride_hailing": "preços ou tempos de espera de Uber, Bolt ou táxi",
    }
    labels = labels_pt if language == "pt" else labels_en
    unsupported_label = _join_transport_labels(
        [labels[mode] for mode in unsupported_modes if mode in labels],
        language,
    )
    supported_links = (
        "[*Metro de Lisboa*](https://www.metrolisboa.pt) | "
        "[*Carris*](https://www.carris.pt) | "
        "[*Carris Metropolitana*](https://www.carrismetropolitana.pt) | "
        "[*CP*](https://www.cp.pt)"
    )
    timestamp = datetime.now().strftime("%H:%M")

    if language == "pt":
        return "\n".join(
            [
                "### ⚠️ Modo de transporte ainda não confirmado neste runtime",
                "",
                f"- Não consigo confirmar {unsupported_label} neste runtime.",
                "- Neste momento, só consigo validar diretamente Metro de Lisboa, Carris Urban, Carris Metropolitana e comboios CP.",
                "- Para evitar alucinações, prefiro não inventar horários, frequências, tarifas, ETAs ou estado em tempo real para esse modo.",
                "- Se a viagem também puder ser formulada com as redes suportadas acima, respondo com base nesses operadores confirmados.",
                "",
                f"📌 **Fonte:** Redes suportadas neste runtime: {supported_links} | **Atualizado:** {timestamp}",
            ]
        ).strip()

    return "\n".join(
        [
            "### ⚠️ Transport mode not yet confirmed in this runtime",
            "",
            f"- I can't directly verify {unsupported_label} in this runtime.",
            "- Right now I can only validate Metro de Lisboa, Carris Urban, Carris Metropolitana, and CP trains.",
            "- To avoid hallucinating details, I won't invent schedules, frequencies, fares, ETAs, or real-time status for that mode.",
            "- If the trip can also be phrased with the supported networks above, I can answer using those confirmed operators.",
            "",
            f"📌 **Source:** Networks supported in this runtime: {supported_links} | **Updated:** {timestamp}",
        ]
    ).strip()


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


def _looks_like_acronym_label(text: str) -> bool:
    """Returns whether the label looks like an acronym that should keep its casing."""
    stripped = (text or "").strip()
    return bool(re.fullmatch(r"(?:[A-Z0-9]{2,}(?:[\s/-][A-Z0-9]{2,})*)", stripped))


def _get_transport_display_name(location: Optional[str], detailed: bool = False) -> str:
    """Returns a user-facing location label while preserving landmark branding like NOVA IMS."""
    from tools.location_resolver import get_location_display_name
    from tools.metrolisboa_api import get_landmark_info

    raw = str(location or "").strip()
    if not raw:
        return raw

    landmark = get_landmark_info(raw)
    if landmark:
        if detailed:
            return str(
                landmark.get("display_name")
                or landmark.get("name")
                or landmark.get("short_name")
                or raw
            ).strip()
        return str(
            landmark.get("short_name")
            or landmark.get("name")
            or landmark.get("display_name")
            or raw
        ).strip()

    if _looks_like_acronym_label(raw):
        return raw

    try:
        resolved_label = get_location_display_name(raw, detailed=detailed)
        if resolved_label:
            return resolved_label
    except Exception:
        pass

    return raw.title()


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
    walk_target_display: Optional[str] = None,
    destination_display: Optional[str] = None,
    destination_landmark: Optional[Dict[str, Any]] = None,
) -> str:
    """Builds a short, practical, non-generic travel tip."""
    if walk_target and final_station and _normalize_token(walk_target) != _normalize_token(final_station):
        walking_hint_pt = (destination_landmark or {}).get("walking_hint_pt")
        walking_hint_en = (destination_landmark or {}).get("walking_hint_en")
        metro_walk_minutes = (destination_landmark or {}).get("metro_walk_minutes")
        destination_label = destination_display or walk_target_display or walk_target

        if language == "pt":
            if walking_hint_pt:
                if metro_walk_minutes:
                    return (
                        f"Se o destino é {destination_label}, saia em {final_station} "
                        f"e conte com cerca de {metro_walk_minutes} min a pé até {walking_hint_pt}."
                    )
                return (
                    f"Se o destino é {destination_label}, saia em {final_station} "
                    f"e conte com uma caminhada final curta até {walking_hint_pt}."
                )
            return f"Da estação {final_station} até {destination_label} a caminhada final é curta."
        if walking_hint_en:
            if metro_walk_minutes:
                return (
                    f"If you're heading to {destination_label}, exit at {final_station} "
                    f"and expect about {metro_walk_minutes} minutes on foot {walking_hint_en}."
                )
            return (
                f"If you're heading to {destination_label}, exit at {final_station} "
                f"and expect a short final walk {walking_hint_en}."
            )
        return f"From {final_station} to {destination_label}, the final walk is short."

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


def _query_mentions_specific_route_mode(user_message: str) -> bool:
    """Returns whether the user explicitly constrained the route to a transport mode."""
    normalized = _normalize_token(user_message)
    return bool(
        re.search(
            r"\b(metro|subway|underground|comboio|comboios|train|trains|cp|autocarro|autocarros|bus|buses|tram|trams|eletrico|eletricos|electrico|electricos)\b",
            normalized,
        )
    )


def _minutes_until_clock_time(clock_text: Optional[str]) -> Optional[int]:
    """Returns minutes from now until the next HH:MM clock occurrence."""
    if not clock_text:
        return None

    target_minutes = _minutes_from_clock(clock_text)
    if target_minutes is None:
        return None

    now = datetime.now()
    current_minutes = now.hour * 60 + now.minute
    diff = target_minutes - current_minutes
    if diff < 0:
        diff += 24 * 60
    return diff


def _build_train_landmark_option(
    origin: str,
    destination: str,
    language: str,
) -> Optional[Tuple[int, str]]:
    """Builds a concise CP alternative for generic route questions when a landmark is near a rail station."""
    from tools.cp_api import get_cp_station_info, plan_train_trip
    from tools.metrolisboa_api import get_landmark_info

    origin_landmark = get_landmark_info(origin)
    destination_landmark = get_landmark_info(destination)
    if not destination_landmark:
        return None

    origin_cp = get_cp_station_info(origin)
    origin_train_station = None
    if origin_landmark and origin_landmark.get("train_station"):
        origin_train_station = str(origin_landmark["train_station"]).strip()
    elif origin_cp:
        origin_train_station = str(origin_cp.get("name") or origin).strip()

    destination_train_station = str(destination_landmark.get("train_station") or "").strip()
    if not origin_train_station or not destination_train_station:
        return None
    if _normalize_token(origin_train_station) == _normalize_token(destination_train_station):
        return None

    try:
        train_result = str(
            plan_train_trip.invoke(
                {"origin": origin_train_station, "destination": destination_train_station}
            )
        )
    except Exception:
        return None

    normalized_result = _normalize_token(train_result)
    if train_result.strip().startswith("❌") or any(
        marker in normalized_result
        for marker in [
            "no direct train service",
            "no more trains today",
            "station not found",
            "error planning trip",
        ]
    ):
        return None

    departure_match = re.search(
        r"🕐\s*\*\*(?P<departure>\d{2}:\d{2})\*\*\s*→\s*(?P<arrival>\d{2}:\d{2})\s*\((?P<travel>\d+)min\)",
        train_result,
    )
    duration_match = re.search(
        r"Dura[cç][aã]o:\s*\*\*(?P<duration>\d+(?:-\d+)?)\s*minutos\*\*",
        train_result,
        flags=re.IGNORECASE,
    )

    departure_clock = departure_match.group("departure") if departure_match else None
    travel_minutes = None
    if departure_match:
        travel_minutes = int(departure_match.group("travel"))
    elif duration_match:
        travel_minutes = int(duration_match.group("duration").split("-", 1)[0])

    walk_minutes = int(destination_landmark.get("train_walk_minutes") or 0)
    wait_minutes = _minutes_until_clock_time(departure_clock)
    score = (wait_minutes or 0) + (travel_minutes or 999) + walk_minutes

    destination_label = _get_transport_display_name(destination)
    if language == "pt":
        summary = f"- 🚆 **Comboio via {destination_train_station}**: "
        if departure_clock:
            summary += f"próxima saída às {departure_clock} desde {origin_train_station}, "
        elif origin_train_station:
            summary += f"saída desde {origin_train_station}, "
        if travel_minutes:
            summary += f"~{travel_minutes} min de viagem"
        else:
            summary += "viagem curta"
        if walk_minutes:
            summary += f" + ~{walk_minutes} min a pé até {destination_label}"
        summary += "."
    else:
        summary = f"- 🚆 **Train via {destination_train_station}**: "
        if departure_clock:
            summary += f"next departure at {departure_clock} from {origin_train_station}, "
        elif origin_train_station:
            summary += f"departing from {origin_train_station}, "
        if travel_minutes:
            summary += f"about {travel_minutes} minutes on board"
        else:
            summary += "a short rail ride"
        if walk_minutes:
            summary += f" + about {walk_minutes} minutes on foot to {destination_label}"
        summary += "."

    return score, summary


def _build_carris_direct_option(
    origin: str,
    destination: str,
    language: str,
) -> Optional[Tuple[int, str]]:
    """Builds a concise direct Carris alternative when one exists."""
    from tools.carris_api import carris_find_routes_between

    try:
        carris_result = str(
            carris_find_routes_between.invoke(
                {"origin": origin, "destination": destination}
            )
        )
    except Exception:
        return None

    normalized_result = _normalize_token(carris_result)
    if any(
        marker in normalized_result
        for marker in [
            "no direct carris route found",
            "could not locate",
            "carris database unavailable",
            "erro ao encontrar rotas",
        ]
    ):
        return None

    buses_match = re.search(
        r"BUSES\n-+\n(?P<section>.+?)(?:\n\n[A-Z]+\n-+|\Z)",
        carris_result,
        flags=re.DOTALL,
    )
    if not buses_match:
        return None

    bus_section = buses_match.group("section")
    route_match = re.search(r"^\s*(?P<route>\d{1,4}[A-Z]?): para (?P<headsign>[^\n]+)", bus_section, flags=re.MULTILINE)
    next_match = re.search(r"^\s*Next: (?P<times>[^\n]+?) \(stop (?P<stop>[^\)]+)\)", bus_section, flags=re.MULTILINE)
    travel_match = re.search(r"^\s*~(?P<travel>\d+)min travel", bus_section, flags=re.MULTILINE)
    if not route_match:
        return None

    route_code = route_match.group("route").strip()
    headsign = route_match.group("headsign").strip()
    stop_name = next_match.group("stop").strip() if next_match else None
    departure_clock = None
    if next_match:
        clock_match = re.search(r"(\d{2}:\d{2})", next_match.group("times"))
        if clock_match:
            departure_clock = clock_match.group(1)
    travel_minutes = int(travel_match.group("travel")) if travel_match else None
    wait_minutes = _minutes_until_clock_time(departure_clock)
    score = (wait_minutes or 0) + (travel_minutes or 999)

    if language == "pt":
        summary = f"- 🚌 **Autocarro {route_code}**: "
        if departure_clock:
            summary += f"próxima saída às {departure_clock}"
        else:
            summary += "rota direta disponível"
        if stop_name:
            summary += f" na paragem {stop_name}"
        if travel_minutes:
            summary += f", ~{travel_minutes} min de viagem"
        summary += f" em direção a {headsign}."
    else:
        summary = f"- 🚌 **Bus {route_code}**: "
        if departure_clock:
            summary += f"next departure at {departure_clock}"
        else:
            summary += "direct route available"
        if stop_name:
            summary += f" from stop {stop_name}"
        if travel_minutes:
            summary += f", about {travel_minutes} minutes travel"
        summary += f" toward {headsign}."

    return score, summary


def _build_additional_route_options(
    user_message: str,
    origin: str,
    destination: str,
    language: str,
) -> List[str]:
    """Builds concise alternative-mode bullets for open-ended route questions."""
    if _query_mentions_specific_route_mode(user_message):
        return []

    options: List[Tuple[int, str]] = []
    train_option = _build_train_landmark_option(origin, destination, language)
    if train_option:
        options.append(train_option)

    bus_option = _build_carris_direct_option(origin, destination, language)
    if bus_option:
        options.append(bus_option)

    if not options:
        return []

    title = (
        "🔁 **Outras opções que também fazem sentido:**"
        if language == "pt"
        else "🔁 **Other sensible options:**"
    )
    sorted_options = [summary for _, summary in sorted(options, key=lambda item: item[0])]
    return ["", title, *sorted_options, ""]


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

    stop_name_arrival_patterns = [
        r"(?:quais\s+)?(?:os\s+)?pr[oó]xim(?:os|as)\s+(?:autocarros?|el[eé]tricos?|autocarros?\s+da\s+carris|ve[ií]culos?\s+da\s+carris|partidas|chegadas)\s+(?:da\s+carris\s+)?(?:at|em|na|no)\s+(?P<stop>.+?)(?:[\?\!\.,;]|$)",
        r"(?:next\s+)?(?:buses?|trams?|arrivals|departures)\s+(?:for\s+carris\s+)?(?:at|in)\s+(?P<stop>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in stop_name_arrival_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {
                "kind": "arrivals",
                "line": None,
                "stop_name": _clean_query_fragment(match.group("stop")),
                "stop_id": stop_id_match.group("stop_id") if stop_id_match else None,
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
    future_planning = _is_future_transport_planning_query(user_message)
    station = request.get("station") or ""
    direction = request.get("direction")
    status_requested = bool(request.get("status_requested"))

    if future_planning:
        return _build_future_metro_wait_limit_response(station, direction, language)

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
    future_planning = _is_future_transport_planning_query(user_message)

    from tools.metrolisboa_api import METRO_LINES, get_landmark_info
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
    origin_display = _get_transport_display_name(endpoints[0])
    destination_display = _get_transport_display_name(endpoints[1])
    destination_detailed_display = _get_transport_display_name(endpoints[1], detailed=True)
    walk_target_display = _get_transport_display_name(walk_target) if walk_target else None
    destination_landmark = get_landmark_info(endpoints[1]) or (
        get_landmark_info(walk_target) if walk_target else None
    )

    if not final_station:
        return None

    first_line_id = _get_line_id_between(board_station, transfer_station or final_station)
    second_line_id = _get_line_id_between(transfer_station, final_station) if transfer_station else None
    line_ids = [line_id for line_id in [first_line_id, second_line_id] if line_id]

    state_lines = [] if future_planning else _build_route_state_lines(line_ids, language)

    wait_targets: List[Tuple[str, str]] = []
    if board_station and first_direction:
        wait_targets.append((board_station, first_direction))
    if transfer_station and second_direction:
        wait_targets.append((transfer_station, second_direction))
    if not wait_targets:
        wait_targets = _parse_wait_targets_from_route(route_result)

    realtime_lines = [] if future_planning else _build_metro_wait_lines(wait_targets, language)

    tip = _build_practical_tip(
        language=language,
        first_direction=first_direction,
        transfer_station=transfer_station,
        second_line_id=second_line_id,
        final_station=final_station,
        walk_target=walk_target,
        walk_target_display=walk_target_display,
        destination_display=destination_detailed_display,
        destination_landmark=destination_landmark,
    )
    additional_options = _build_additional_route_options(
        user_message=user_message,
        origin=endpoints[0],
        destination=endpoints[1],
        language=language,
    )

    route_title = f"🚇 **{origin_display}** → **{destination_display}**"
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

    response_lines = [route_title, ""]

    if state_lines:
        response_lines.extend([
            state_title,
            *state_lines,
            "",
        ])

    response_lines.extend([
        f"{time_title} {estimated_time}",
        "",
        route_section,
        f"{board_text} {board_station}**",
    ])

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
        response_lines.append(f"{walk_text} {walk_target_display or walk_target}**")

    if future_planning:
        response_lines.extend([
            "",
            _build_future_realtime_note(language),
            "",
        ])
    else:
        response_lines.extend([
            "",
            waits_section,
            *realtime_lines,
            "",
        ])

    if tip:
        response_lines.append(f"{tip_title} {tip}")
        response_lines.append("")

    if additional_options:
        response_lines.extend(additional_options)

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
    if _query_has_route_mode_constraints(query):
        return None

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
        self._last_transport_context: Optional[dict] = None

    def reset_conversation_context(self) -> None:
        """Clears cached transport follow-up context for the session."""
        self._last_transport_context = None

    def get_last_transport_context(self) -> Optional[dict]:
        """Returns the latest cached transport follow-up context."""
        return deepcopy(self._last_transport_context)

    def _get_tool_by_name(self, tool_name: str):
        """Returns a loaded tool by name, or None if not found."""
        for tool in self.tools:
            if getattr(tool, "name", "") == tool_name:
                return tool
        return None

    @staticmethod
    def _extract_follow_up_mode(user_message: str) -> Optional[str]:
        """Extracts a requested transport mode from a short follow-up."""
        query = (user_message or "").lower()
        if any(term in query for term in ["metro"]):
            return "metro"
        if any(term in query for term in ["bus", "autocarro", "autocarros", "carris"]):
            return "bus"
        if any(term in query for term in ["train", "comboio", "comboios", "cp"]):
            return "train"
        if any(term in query for term in ["tram", "trams", "elétrico", "eletrico", "elétricos", "eletricos"]):
            return "tram"
        return None

    def _rewrite_follow_up_transport_query(self, user_message: str, language: str) -> str:
        """Rewrites short transport follow-ups using the last remembered route endpoints."""
        if _extract_route_endpoints(user_message):
            return user_message

        normalized = re.sub(r"[!?.,;:]+", "", (user_message or "").strip().lower())
        follow_up_prefixes = ("e ", "and ", "what about", "how about", "same ", "also ", "agora ", "now ")
        is_short_referential_follow_up = normalized.startswith(follow_up_prefixes) or len(normalized.split()) <= 6
        if not is_short_referential_follow_up:
            return user_message

        last_context = getattr(self, "_last_transport_context", None)
        if not last_context:
            return user_message

        mode = self._extract_follow_up_mode(user_message)
        if not mode:
            return user_message

        origin = str(last_context.get("origin") or "").strip()
        destination = str(last_context.get("destination") or "").strip()
        if not origin or not destination:
            return user_message

        if language == "pt":
            mode_phrase = {
                "metro": "de metro",
                "bus": "de autocarro",
                "train": "de comboio",
                "tram": "de elétrico",
            }.get(mode, "")
            return f"Como vou de {origin} para {destination} {mode_phrase}?".strip()

        mode_phrase = {
            "metro": "by metro",
            "bus": "by bus",
            "train": "by train",
            "tram": "by tram",
        }.get(mode, "")
        return f"How do I get from {origin} to {destination} {mode_phrase}?".strip()

    def _remember_transport_context(self, user_message: str) -> None:
        """Caches the latest route endpoints so transport-mode follow-ups can reuse them."""
        endpoints = _extract_route_endpoints(user_message)
        if not endpoints:
            return

        self._last_transport_context = {
            "origin": endpoints[0],
            "destination": endpoints[1],
            "last_user_query": user_message,
            "mode": self._extract_follow_up_mode(user_message),
        }

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
                return self._invoke_tool(metro_tool, {}, tool_name="get_metro_status")

        summary_tool = self._get_tool_by_name("get_transport_summary")
        if summary_tool:
            return self._invoke_tool(summary_tool, {}, tool_name="get_transport_summary")

        return None

    @staticmethod
    def _build_transport_source_line(language: str, source_links: List[str]) -> str:
        """Builds a localized transport source line for combined deterministic answers."""
        deduped_sources: List[str] = []
        for source in source_links:
            if source and source not in deduped_sources:
                deduped_sources.append(source)

        timestamp = datetime.now().strftime("%H:%M")
        if language == "pt":
            return f"📌 **Fonte:** {' | '.join(deduped_sources)} | **Atualizado:** {timestamp}"
        return f"📌 **Source:** {' | '.join(deduped_sources)} | **Updated:** {timestamp}"

    @staticmethod
    def _extract_train_lines_summary(summary_text: str) -> Optional[str]:
        """Extracts the CP line summary from a train-tool response."""
        if not summary_text:
            return None

        match = re.search(r"(?:Linhas?|Line(?:s)?):\s*\**([^*\n]+)\**", summary_text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _is_mode_comparison_query(user_message: str) -> bool:
        """Detect explicit transport-mode comparison requests such as metro vs train."""
        normalized_query = unicodedata.normalize("NFKD", user_message or "")
        normalized_query = normalized_query.encode("ascii", "ignore").decode("ascii").lower()
        compares_metro_train = (
            "metro" in normalized_query
            and any(term in normalized_query for term in ["comboio", "comboios", "train", "trains"])
        )
        asks_comparison = bool(
            re.search(r"\b(mais rapid[oa]|mais barat[oa]|faster|fastest|cheaper|cheapest|compare|comparar)\b", normalized_query)
        )
        return compares_metro_train and asks_comparison

    @staticmethod
    def _extract_duration_minutes(summary_text: str) -> Optional[int]:
        """Extract a best-effort trip duration in minutes from a transport summary."""
        if not summary_text:
            return None

        normalized_text = unicodedata.normalize("NFKD", summary_text)
        normalized_text = normalized_text.encode("ascii", "ignore").decode("ascii")
        patterns = [
            r"(?:Tempo total estimado|Estimated total time|Duracao|Duration):\s*\**~?\s*(\d+)",
            r"~\s*(\d+)\s*min",
            r"\b(\d+)\s*min(?:utos?)?\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized_text, flags=re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _extract_departure_times(summary_text: str, limit: int = 3) -> List[str]:
        """Extract the first distinct departure times listed in a summary."""
        if not summary_text:
            return []

        matches = re.findall(r"(?:\*\*)?(\d{1,2}:\d{2})(?:\*\*)?\s*(?:→|>)", summary_text)
        deduped: List[str] = []
        for item in matches:
            if item not in deduped:
                deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def _build_mode_comparison_response(
        self,
        user_message: str,
        context: str = "",
        language: str = "en",
    ) -> Optional[str]:
        """Build a deterministic metro-vs-train comparison answer when the user asks for one."""
        if not self._is_mode_comparison_query(user_message):
            return None

        endpoints = _extract_route_endpoints(user_message)
        if not endpoints:
            return None

        origin, destination = endpoints
        metro_response = _build_deterministic_metro_route_response(user_message, context) or ""

        train_tool = self._get_tool_by_name("plan_train_trip")
        train_response = ""
        if train_tool:
            train_response = str(
                self._invoke_tool(
                    train_tool,
                    {"origin": origin, "destination": destination},
                    tool_name="plan_train_trip",
                )
            ).strip()

        if not metro_response and not train_response:
            return None

        metro_minutes = self._extract_duration_minutes(metro_response)
        train_minutes = self._extract_duration_minutes(train_response)
        train_lines = self._extract_train_lines_summary(train_response)
        train_departures = self._extract_departure_times(train_response, limit=3)
        normalized_query = unicodedata.normalize("NFKD", user_message or "")
        normalized_query = normalized_query.encode("ascii", "ignore").decode("ascii").lower()
        asks_cheapest = bool(re.search(r"\b(mais barat[oa]|cheaper|cheapest|preco|price)\b", normalized_query))

        arrow = "\u2192"
        if language == "pt":
            lines = [
                f"**Compara\u00e7\u00e3o:** {origin} {arrow} {destination}",
                "",
                "#### \U0001F687 Metro",
                f"\u23F1\uFE0F **Tempo estimado:** {metro_minutes} min"
                if metro_minutes is not None
                else "\u23F1\uFE0F **Tempo estimado:** n\u00e3o foi poss\u00edvel confirmar com os dados dispon\u00edveis",
                "",
                "#### \U0001F686 Comboio",
                f"\u23F1\uFE0F **Tempo estimado:** {train_minutes} min"
                if train_minutes is not None
                else "\u23F1\uFE0F **Tempo estimado:** n\u00e3o foi poss\u00edvel confirmar com os dados dispon\u00edveis",
            ]
            if train_lines:
                lines.append(f"\U0001F686 **Linhas:** {train_lines}")
            if train_departures:
                lines.append(f"\U0001F550 **Pr\u00f3ximas sa\u00eddas mostradas:** {', '.join(train_departures)}")

            lines.extend(["", "#### \u2705 Conclus\u00e3o"])
            if metro_minutes is not None and train_minutes is not None:
                faster_label = "Comboio" if train_minutes < metro_minutes else "Metro"
                lines.append(f"- **Mais r\u00e1pido:** {faster_label}")
            else:
                lines.append("- **Mais r\u00e1pido:** n\u00e3o foi poss\u00edvel comparar com seguran\u00e7a porque falta pelo menos uma dura\u00e7\u00e3o oficial")
            if asks_cheapest:
                lines.append("- **Mais barato:** n\u00e3o foi poss\u00edvel confirmar com dados oficiais de tarifa nas tools dispon\u00edveis")
        else:
            lines = [
                f"**Comparison:** {origin} {arrow} {destination}",
                "",
                "#### \U0001F687 Metro",
                f"\u23F1\uFE0F **Estimated time:** {metro_minutes} min"
                if metro_minutes is not None
                else "\u23F1\uFE0F **Estimated time:** could not be confirmed from the available data",
                "",
                "#### \U0001F686 Train",
                f"\u23F1\uFE0F **Estimated time:** {train_minutes} min"
                if train_minutes is not None
                else "\u23F1\uFE0F **Estimated time:** could not be confirmed from the available data",
            ]
            if train_lines:
                lines.append(f"\U0001F686 **Lines:** {train_lines}")
            if train_departures:
                lines.append(f"\U0001F550 **Next departures shown:** {', '.join(train_departures)}")

            lines.extend(["", "#### \u2705 Verdict"])
            if metro_minutes is not None and train_minutes is not None:
                faster_label = "Train" if train_minutes < metro_minutes else "Metro"
                lines.append(f"- **Faster:** {faster_label}")
            else:
                lines.append("- **Faster:** I could not compare confidently because at least one official duration is missing")
            if asks_cheapest:
                lines.append("- **Cheaper:** official fare data could not be confirmed with the currently available tools")

        lines.extend([
            "",
            self._build_transport_source_line(
                language,
                [
                    "[*Metro de Lisboa*](https://www.metrolisboa.pt)",
                    "[*CP*](https://www.cp.pt)",
                ],
            ),
        ])
        return "\n".join(lines).strip()

    def _format_deterministic_tool_result(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: str,
        language: str,
    ) -> str:
        """Post-process deterministic single-tool outputs for cleaner user rendering."""
        cleaned_result = str(result or "").strip()
        if not cleaned_result:
            return cleaned_result

        if tool_name == "find_direct_bus_lines":
            filtered_lines: List[str] = []
            for raw_line in cleaned_result.splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    continue
                lowered = stripped.lower()
                if (
                    (stripped.startswith("⚠️") and "scope" in lowered)
                    or stripped.startswith(("💡 For Lisbon city-only", "📌 **Source:", "📌 **Fonte:"))
                ):
                    continue

                line = raw_line.rstrip()
                if language != "pt":
                    replacements = {
                        "Autocarros": "Buses",
                        "linha(s) direta(s) encontrada(s)": "direct line(s) found",
                        "Terminais": "Terminals",
                        "Outras linhas": "Other lines",
                        "Como usar": "How to use it",
                        "Procure pelo número da linha": "Look for the line number",
                        "Verifique a direção do autocarro": "Check the bus direction",
                        "Horários e paragens": "Schedules and stops",
                        "na paragem": "at the stop",
                        " (ex: ": " (e.g. ",
                    }
                    for old, new in replacements.items():
                        line = line.replace(old, new)

                filtered_lines.append(line)

            if filtered_lines:
                origin = str(tool_args.get("origin") or "Origin").strip()
                destination = str(tool_args.get("destination") or "Destination").strip()
                title = (
                    f"### 🚌 Linhas diretas da Carris Metropolitana para {origin} → {destination}"
                    if language == "pt"
                    else f"### 🚌 Direct Carris Metropolitana lines for {origin} → {destination}"
                )
                return "\n".join(
                    [
                        title,
                        "",
                        *filtered_lines,
                        "",
                        self._build_transport_source_line(
                            language,
                            ["[*Carris Metropolitana*](https://www.carrismetropolitana.pt)"],
                        ),
                    ]
                ).strip()

        return cleaned_result

    def _build_mode_constrained_route_response(
        self,
        user_message: str,
        context: str = "",
        language: str = "en",
    ) -> Optional[str]:
        """Builds deterministic responses for route queries with explicit mode constraints."""
        endpoints = _extract_route_endpoints(user_message)
        if not endpoints:
            return None

        preferences = _parse_route_mode_preferences(user_message)
        if not any(preferences.values()):
            return None

        origin, destination = endpoints
        language = language or infer_response_language(user_query=user_message, default="en")

        if preferences["metro_only"] or (
            preferences["exclude_bus"]
            and not preferences["tram_only"]
            and not preferences["exclude_metro"]
        ):
            metro_response = _build_deterministic_metro_route_response(
                user_message=user_message,
                context=context,
            )
            if metro_response:
                return metro_response

        urban_tool = self._get_tool_by_name("carris_find_routes_between")
        frequency_tool = self._get_tool_by_name("carris_get_service_frequency")
        urban_result = (
            str(
                self._invoke_tool(
                    urban_tool,
                    {"origin": origin, "destination": destination},
                    tool_name="carris_find_routes_between",
                )
            ).strip()
            if urban_tool
            else ""
        )

        def frequency_lookup(route_short_name: str) -> str:
            if not frequency_tool or not route_short_name:
                return ""
            try:
                return str(
                    self._invoke_tool(
                        frequency_tool,
                        {"route_short_name": route_short_name},
                        tool_name="carris_get_service_frequency",
                    )
                ).strip()
            except Exception:
                return ""

        if preferences["bus_only"] or preferences["exclude_tram"]:
            urban_bus_block = _extract_carris_mode_section(urban_result, "BUSES")
            urban_tram_block = _extract_carris_mode_section(urban_result, "TRAMS")
            metropolitan_tool = self._get_tool_by_name("find_direct_bus_lines")
            metropolitan_result = (
                str(
                    self._invoke_tool(
                        metropolitan_tool,
                        {"origin": origin, "destination": destination},
                        tool_name="find_direct_bus_lines",
                    )
                ).strip()
                if metropolitan_tool
                else ""
            )
            urban_bus_markdown = _format_carris_mode_section_markdown(
                urban_bus_block,
                language,
                frequency_lookup=frequency_lookup,
            )
            metropolitan_block = _clean_metropolitana_direct_bus_block(metropolitan_result)

            sections: List[str] = []
            notes: List[str] = []

            if urban_bus_markdown:
                sections.append(f"#### 🚌 Carris Urban\n\n{urban_bus_markdown}")
            elif _carris_section_has_routes(urban_tram_block):
                notes.append(
                    "- **Carris Urban:** only tram options were found for this trip, not bus-only ones."
                    if language != "pt"
                    else "- **Carris Urban:** só apareceram opções de elétrico nesta ligação, não opções apenas de autocarro."
                )
            elif urban_result and not _tool_result_indicates_no_match(urban_result):
                notes.append(
                    "- **Carris Urban:** no bus-only route could be isolated from the available urban result."
                    if language != "pt"
                    else "- **Carris Urban:** não foi possível isolar uma rota apenas de autocarro no resultado urbano disponível."
                )

            if metropolitan_result and not _tool_result_indicates_no_match(metropolitan_result):
                if metropolitan_block:
                    sections.append(
                        f"#### 🚌 Carris Metropolitana\n\n{metropolitan_block}"
                    )
            else:
                notes.append(
                    "- **Carris Metropolitana:** no direct suburban bus line was confirmed for this trip."
                    if language != "pt"
                    else "- **Carris Metropolitana:** não foi confirmada nenhuma linha suburbana direta para esta ligação."
                )

            if not sections:
                if language == "pt":
                    message = (
                        f"❌ Não consegui confirmar uma rota apenas de autocarro entre {origin} e {destination} com os dados disponíveis da Carris Urban e da Carris Metropolitana."
                    )
                else:
                    message = (
                        f"❌ I couldn't confirm a bus-only route between {origin} and {destination} with the available Carris Urban and Carris Metropolitana data."
                    )
                if notes:
                    notes_title = "#### ℹ️ Coverage notes" if language != "pt" else "#### ℹ️ Notas de cobertura"
                    message += "\n\n" + notes_title + "\n" + "\n".join(notes)
                message += "\n\n" + self._build_transport_source_line(
                    language,
                    [
                        "[*Carris*](https://www.carris.pt)",
                        "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
                    ],
                )
                return message

            intro = (
                f"### 🚌 Bus-only options for {origin} → {destination}"
                if language != "pt"
                else f"### 🚌 Opções apenas de autocarro para {origin} → {destination}"
            )
            response_parts = [intro, "", *sections]
            if notes:
                notes_title = "#### ℹ️ Coverage notes" if language != "pt" else "#### ℹ️ Notas de cobertura"
                response_parts.extend(["", notes_title, *notes])
            response_parts.extend(
                [
                    "",
                    self._build_transport_source_line(
                        language,
                        [
                            "[*Carris*](https://www.carris.pt)",
                            "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
                        ],
                    ),
                ]
            )
            return "\n".join(response_parts).strip()

        if preferences["exclude_metro"]:
            urban_bus_block = _extract_carris_mode_section(urban_result, "BUSES")
            urban_tram_block = _extract_carris_mode_section(urban_result, "TRAMS")
            metropolitan_tool = self._get_tool_by_name("find_direct_bus_lines")
            metropolitan_result = (
                str(
                    self._invoke_tool(
                        metropolitan_tool,
                        {"origin": origin, "destination": destination},
                        tool_name="find_direct_bus_lines",
                    )
                ).strip()
                if metropolitan_tool
                else ""
            )

            urban_bus_markdown = _format_carris_mode_section_markdown(
                urban_bus_block,
                language,
                frequency_lookup=frequency_lookup,
            )
            urban_tram_markdown = _format_carris_mode_section_markdown(
                urban_tram_block,
                language,
                frequency_lookup=frequency_lookup,
            )
            metropolitan_block = _clean_metropolitana_direct_bus_block(metropolitan_result)

            sections: List[str] = []
            if urban_bus_markdown:
                sections.append(f"#### 🚌 Carris Urban\n\n{urban_bus_markdown}")
            if urban_tram_markdown:
                sections.append(f"#### 🚋 Carris Urban\n\n{urban_tram_markdown}")
            if metropolitan_result and not _tool_result_indicates_no_match(metropolitan_result) and metropolitan_block:
                sections.append(f"#### 🚌 Carris Metropolitana\n\n{metropolitan_block}")

            if not sections:
                if language == "pt":
                    return (
                        f"❌ Não consegui confirmar uma rota sem metro entre {origin} e {destination} com os dados de superfície disponíveis.\n\n"
                        + self._build_transport_source_line(
                            language,
                            [
                                "[*Carris*](https://www.carris.pt)",
                                "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
                            ],
                        )
                    )
                return (
                    f"❌ I couldn't confirm a non-metro surface route between {origin} and {destination} with the available data.\n\n"
                    + self._build_transport_source_line(
                        language,
                        [
                            "[*Carris*](https://www.carris.pt)",
                            "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
                        ],
                    )
                )

            intro = (
                f"### 🚌🚋 Surface options without metro for {origin} → {destination}"
                if language != "pt"
                else f"### 🚌🚋 Opções de superfície sem metro para {origin} → {destination}"
            )
            return "\n".join(
                [
                    intro,
                    "",
                    *sections,
                    "",
                    self._build_transport_source_line(
                        language,
                        [
                            "[*Carris*](https://www.carris.pt)",
                            "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
                        ],
                    ),
                ]
            ).strip()

        if preferences["tram_only"] or preferences["exclude_bus"]:
            urban_tram_block = _extract_carris_mode_section(urban_result, "TRAMS")
            if _carris_section_has_routes(urban_tram_block):
                urban_tram_markdown = _format_carris_mode_section_markdown(
                    urban_tram_block,
                    language,
                    frequency_lookup=frequency_lookup,
                )
                intro = (
                    f"### 🚋 Tram-only options for {origin} → {destination}"
                    if language != "pt"
                    else f"### 🚋 Opções apenas de elétrico para {origin} → {destination}"
                )
                return "\n".join(
                    [
                        intro,
                        "",
                        "#### 🚋 Carris Urban",
                        "",
                        urban_tram_markdown or urban_tram_block,
                        "",
                        self._build_transport_source_line(
                            language,
                            ["[*Carris*](https://www.carris.pt)"],
                        ),
                    ]
                ).strip()

            if language == "pt":
                return (
                    f"❌ Não consegui confirmar uma rota sem autocarro entre {origin} e {destination} com os dados disponíveis.\n\n"
                    f"📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | [*Carris*](https://www.carris.pt) | **Atualizado:** {datetime.now().strftime('%H:%M')}"
                )
            return (
                f"❌ I couldn't confirm a non-bus route between {origin} and {destination} with the available data.\n\n"
                f"📌 **Source:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | [*Carris*](https://www.carris.pt) | **Updated:** {datetime.now().strftime('%H:%M')}"
            )

        return None

    def _resolve_deterministic_response(
        self,
        user_message: str,
        context: str = "",
        language: Optional[str] = None,
    ) -> Optional[str]:
        """Resolves deterministic direct-response fast paths shared by invoke() and build_subgraph()."""
        resolved_language = language or infer_response_language(user_query=user_message, default="en")

        unsupported_mode_response = _build_unsupported_transport_scope_response(
            user_message=user_message,
            language=resolved_language,
        )
        if unsupported_mode_response:
            return finalize_worker_response(
                unsupported_mode_response,
                agent_name="transport",
                user_query=user_message,
                language=resolved_language,
            )

        comparison_response = self._build_mode_comparison_response(
            user_message=user_message,
            context=context,
            language=resolved_language,
        )
        if comparison_response:
            return finalize_worker_response(
                comparison_response,
                agent_name="transport",
                user_query=user_message,
                language=resolved_language,
            )

        constrained_route_response = self._build_mode_constrained_route_response(
            user_message=user_message,
            context=context,
            language=resolved_language,
        )
        if constrained_route_response:
            return finalize_worker_response(
                constrained_route_response,
                agent_name="transport",
                user_query=user_message,
                language=resolved_language,
            )

        deterministic_response = _build_deterministic_metro_route_response(
            user_message=user_message,
            context=context,
        )
        if deterministic_response:
            endpoints = _extract_route_endpoints(user_message)
            if endpoints:
                self._record_tool_call(
                    "get_route_between_stations",
                    {"origin": endpoints[0], "destination": endpoints[1]},
                )
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
            metro_wait_request = _parse_metro_wait_request(user_message)
            if metro_wait_request:
                wait_args = {"station": metro_wait_request.get("station")}
                if metro_wait_request.get("direction"):
                    wait_args["direction"] = metro_wait_request.get("direction")
                self._record_tool_call("get_metro_wait_time", wait_args)
            return finalize_worker_response(
                metro_wait_response,
                agent_name="transport",
                user_query=user_message,
                language=resolved_language,
            )

        carris_stop_response = _build_deterministic_carris_stop_response(user_message)
        if carris_stop_response:
            carris_stop_request = _parse_carris_line_stop_query(user_message)
            if carris_stop_request:
                kind = carris_stop_request.get("kind")
                line = carris_stop_request.get("line")
                stop_reference = carris_stop_request.get("stop_id") or carris_stop_request.get("stop_name")
                if kind == "arrivals":
                    self._record_tool_call(
                        "carris_get_arrivals",
                        {"stop": stop_reference, "limit": 8},
                    )
                elif kind == "departures":
                    self._record_tool_call(
                        "carris_get_next_departures",
                        {"route_short_name": line, "stop": stop_reference},
                    )
                elif kind == "eta":
                    self._record_tool_call(
                        "carris_vehicle_eta",
                        {"route_short_name": line, "stop": stop_reference},
                    )
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

        result = self._invoke_tool(tool, tool_args, tool_name=tool_name)
        resolved_language = language or infer_response_language(user_query=user_message, default="en")
        formatted_result = self._format_deterministic_tool_result(
            tool_name=tool_name,
            tool_args=tool_args,
            result=str(result).strip(),
            language=resolved_language,
        )
        return finalize_worker_response(
            formatted_result,
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

        if _is_future_transport_planning_query(user_message):
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
        # Extract explicit language preference from context if provided
        import re
        language_match = re.search(r"User language:\s*(en|pt)", context, re.IGNORECASE)
        if language_match:
            language = language_match.group(1).lower()
        else:
            language = infer_response_language(user_query=user_message, default="en")
        effective_user_message = self._rewrite_follow_up_transport_query(user_message, language)
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

        messages.append(HumanMessage(content=effective_user_message))

        # Skip tool enforcement for greetings/thanks
        is_greeting = any(
            w in effective_user_message.lower()
            for w in ["hello", "thanks", "obrigado", "tchau", "olá", "bom dia"]
        )

        if not is_greeting:
            deterministic_response = self._resolve_deterministic_response(
                user_message=effective_user_message,
                context=context,
                language=language,
            )
            if deterministic_response:
                self._remember_transport_context(effective_user_message)
                return deterministic_response

            deterministic_tool_response = self._invoke_deterministic_tool_call(
                user_message=effective_user_message,
                language=language,
            )
            if deterministic_tool_response:
                self._remember_transport_context(effective_user_message)
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

        self._remember_transport_context(effective_user_message)
        return finalize_worker_response(
            self._ensure_realtime_wait_times(effective_user_message, response),
            agent_name="transport",
            user_query=effective_user_message,
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
                        result = self._invoke_tool(tool, tool_args, tool_name=tool_name)
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
    import os

    counters = {"passed": 0, "failed": 0}

    def _check(condition: bool, label: str) -> None:
        if condition:
            counters["passed"] += 1
            print(f"   \033[1;32m✅ PASS\033[0m: {label}")
        else:
            counters["failed"] += 1
            print(f"   \033[1;31m❌ FAIL\033[0m: {label}")

    deterministic_cases = {
        "Is the metro working?": "get_metro_status",
        "When are the next trains from Entrecampos?": "get_train_schedule",
        "How do I get from Rossio to Sintra by train?": "plan_train_trip",
        "Show real-time Carris Metropolitana buses near Almada": "get_real_time_bus_positions",
        "What are the direct Carris Metropolitana buses from Oeiras to Amadora?": "find_direct_bus_lines",
    }

    print("\n\033[1m📝 Offline deterministic smoke checks:\033[0m")
    for query, expected_tool in deterministic_cases.items():
        deterministic_call = _build_deterministic_transport_tool_call(query)
        resolved_tool = None
        if deterministic_call and deterministic_call.tool_calls:
            resolved_tool = deterministic_call.tool_calls[0].get("name")
            print(f"   🔧 {query} -> {resolved_tool}")
        _check(resolved_tool == expected_tool, f"Deterministic routing selects {expected_tool} for '{query}'")

    parsed_carris_query = _parse_carris_line_stop_query("What are the next departures for 732 at Rossio?")
    _check(
        isinstance(parsed_carris_query, dict)
        and parsed_carris_query.get("kind") == "departures"
        and parsed_carris_query.get("line") == "732",
        "Carris stop/line parser extracts departures queries correctly",
    )
    _check(
        _is_future_transport_planning_query("Como vou amanhã do Rossio ao Aeroporto de metro?") is True,
        "Future transport planning detector recognises tomorrow-style queries",
    )

    formatting_agent = TransportAgent.__new__(TransportAgent)
    formatted_cm_output = formatting_agent._format_deterministic_tool_result(
        tool_name="find_direct_bus_lines",
        tool_args={"origin": "Oeiras", "destination": "Amadora"},
        result=(
            "🚌 **Buses: Oeiras → Amadora**\n\n"
            "✅ **19 direct line(s) found:**\n\n"
            "**1. 🚍 Linha 1501**\n"
            " 📍 **Terminals**: Reboleira (Estação) | Circular via Alfragide\n"
            "💡 **How to use it:**\n"
            " - Look for the line number\n"
            "⚠️ **Scope**: raw wrapper that should disappear"
        ),
        language="en",
    )
    _check(formatted_cm_output.startswith("### 🚌 Direct Carris Metropolitana lines"), "Carris Metropolitana formatter adds a clean title")
    _check("Scope" not in formatted_cm_output, "Carris Metropolitana formatter strips raw scope lines")
    _check("How to use it" in formatted_cm_output, "Carris Metropolitana formatter preserves helpful usage hints")

    future_wait_response = _build_future_metro_wait_limit_response("Saldanha", "Odivelas", "pt")
    _check("tempo real" in future_wait_response.lower(), "Future metro wait response explains the real-time limitation")

    if os.getenv("LISBOA_RUN_LIVE_TRANSPORT_TESTS") == "1":
        print("\n\033[1m🌐 Optional live transport smoke:\033[0m")
        try:
            live_agent = TransportAgent()
            for query in [
                "Is the metro working?",
                "What are the direct Carris Metropolitana buses from Oeiras to Amadora?",
            ]:
                live_response = live_agent.invoke(query)
                print(f"\n{query}\n{'-' * len(query)}")
                print(live_response[:1200])
                _check(bool(live_response.strip()), f"Live transport smoke returned content for '{query}'")
        except Exception as exc:
            _check(False, f"Live transport smoke failed: {exc}")
    else:
        print("\n   ℹ️ Live transport smoke skipped. Set LISBOA_RUN_LIVE_TRANSPORT_TESTS=1 to enable it.")

    print(f"\n\033[1mSummary:\033[0m Passed={counters['passed']} Failed={counters['failed']}")
    if counters["failed"]:
        raise SystemExit(1)
    print("\n\033[1;32m✅ Transport agent smoke test passed!\033[0m")
