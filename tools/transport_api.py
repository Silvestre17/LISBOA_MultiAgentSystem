# ==========================================================================
# Master Thesis - Multi-Modal Transport API Tools
#   - André Filipe Gomes Silvestre, 20240502
#
#   Multi-modal routing and transport summary for Lisbon Metropolitan Area.
#   This module combines data from:
#     - Metro de Lisboa (metrolisboa_api.py)
#     - Carris Metropolitana (carrismetropolitana_api.py)
#     - CP Trains (cp_api.py)
#     - Carris Urban (carris_api.py)
#
#   For individual transport APIs, use the specific modules:
#     - tools.metrolisboa_api: Metro stations, wait times, status
#     - tools.carrismetropolitana_api: Suburban bus routes, stops, alerts
#     - tools.cp_api: Train status, stations, delays
#     - tools.carris_api: Urban Lisbon buses and trams
#
#   Usage:
#     > python tools/transport_api.py
#       Run the manual multi-modal transport integration test suite.
# ==========================================================================

# Required libraries:
# pip install requests langchain-core

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

try:
    import config as _project_config
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
else:
    del _project_config

from tools.cp_api import (
    CP_LINES,
    get_cp_aml_trains,
    get_cp_station_info,
)
from tools.location_resolver import get_location_display_name, normalize_location_text
from tools.utils import haversine_distance

# Import from the split modules
from tools.metrolisboa_api import (
    METRO_LINES,
    fetch_json_with_retry,
    get_landmark_info,
    get_station_lines,
)

logger = logging.getLogger(__name__)

# Metro fallback URL
METRO_STATUS_URL = "https://app.metrolisboa.pt/status/getLinhas.php"

# User-facing interchanges are often named after the rail/bus hub rather than
# the official Metro station. Keep these aliases canonical for route math.
_METRO_STATION_ALIASES: Dict[str, str] = {
    "sete rios": "jardim zoológico",
    "terminal sete rios": "jardim zoológico",
    "marques pombal": "Marquês de Pombal",
    "marques de pombal": "Marquês de Pombal",
    "rotunda do marques": "Marquês de Pombal",
    "rotunda marques": "Marquês de Pombal",
}


# ==========================================================================
# Helper Functions
# ==========================================================================

def _normalize_station(text: str) -> str:
    """Normalizes station or place text for comparison."""
    return normalize_location_text(text)


def _canonical_metro_station_name(station_name: str) -> str:
    """Return the official Metro station name for known user-facing aliases."""
    raw = str(station_name or "").strip()
    return _METRO_STATION_ALIASES.get(_normalize_station(raw), raw)


def _format_location_display_name(location: str, detailed: bool = False) -> str:
    """Formats route endpoints and landmark labels without breaking acronyms like NOVA IMS."""
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

    if re.fullmatch(r"(?:[A-Z0-9]{2,}(?:[\s/-][A-Z0-9]{2,})*)", raw):
        return raw

    try:
        resolved_label = get_location_display_name(raw, detailed=detailed)
        if resolved_label:
            return resolved_label
    except Exception as exc:
        logger.debug("Display-name resolution failed for '%s': %s", raw, exc)

    return raw.title()


def find_nearest_stops_for_place(
    place_name: str,
    max_results: int = 3,
    max_radius_km: float = 0.8,
) -> Dict[str, Any]:
    """Resolve a place to nearby metro, train, and Carris Urban stops using coordinates."""
    try:
        from tools.carris_api import _get_db_connection, geocode_location
    except ImportError:
        return {}

    lat, lon, display_name = geocode_location(place_name)
    if lat is None or lon is None:
        try:
            from tools.location_resolver import geocode_location_name
        except ImportError:
            geocoded = None
        else:
            geocoded = geocode_location_name(place_name, prefer_city=True, allow_aml=True)

        if geocoded:
            lat = geocoded.get("lat")
            lon = geocoded.get("lon")
            display_name = geocoded.get("display_name") or geocoded.get("full_display_name") or display_name

    if lat is None or lon is None:
        return {}

    landmark = get_landmark_info(place_name) or {}
    nearby_stops: List[Dict[str, Any]] = []
    connection = _get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT stop_id, stop_name, stop_lat, stop_lon FROM stops")
            rows = cursor.fetchall()
            search_radii = [max_radius_km, 1.2, 1.8]
            for radius in search_radii:
                candidates: List[Dict[str, Any]] = []
                for row in rows:
                    stop_lat = row["stop_lat"]
                    stop_lon = row["stop_lon"]
                    if stop_lat is None or stop_lon is None:
                        continue

                    distance_km = haversine_distance(float(lat), float(lon), float(stop_lat), float(stop_lon))
                    if distance_km > radius:
                        continue

                    candidates.append(
                        {
                            "stop_id": row["stop_id"],
                            "stop_name": row["stop_name"],
                            "distance_km": distance_km,
                        }
                    )

                if candidates:
                    nearby_stops = sorted(candidates, key=lambda item: (item["distance_km"], item["stop_name"]))
                    break
        finally:
            connection.close()

    return {
        "query": place_name,
        "display_name": (
            landmark.get("display_name")
            or landmark.get("name")
            or landmark.get("short_name")
            or display_name
            or place_name
        ),
        "metro": landmark.get("metro"),
        "metro_line": landmark.get("line", ""),
        "metro_walk_minutes": landmark.get("metro_walk_minutes"),
        "train_station": landmark.get("train_station"),
        "train_walk_minutes": landmark.get("train_walk_minutes"),
        "carris_stops": nearby_stops[:max_results],
    }


def _find_station_index(stations: list, station_name: str) -> int:
    """
    Finds the index of a station in an ordered line list.
    Uses fuzzy matching with accent normalization.

    Args:
        stations: Ordered list of station names on a line.
        station_name: Station name to find.

    Returns:
        Index of station, or -1 if not found.
    """
    name_norm = _normalize_station(station_name)

    # Exact match first
    for i, s in enumerate(stations):
        if _normalize_station(s) == name_norm:
            return i

    # Partial match
    for i, s in enumerate(stations):
        s_norm = _normalize_station(s)
        if name_norm in s_norm or s_norm in name_norm:
            return i

    return -1


def _get_metro_direction(line_id: str, start: str, end: str) -> str:
    """Helper to determine direction (terminal station) on a Metro line."""
    stations = METRO_LINES.get(line_id, {}).get("stations", [])
    if not stations:
        return ""

    idx_start = _find_station_index(stations, start)
    idx_end = _find_station_index(stations, end)

    if idx_start < 0 or idx_end < 0:
        return ""

    if idx_start < idx_end:
        return f"→ direção **{stations[-1].title()}**"
    else:
        return f"→ direção **{stations[0].title()}**"


def _count_metro_stations(line_id: str, start: str, end: str) -> int:
    """
    Counts the number of stations between two points on a Metro line.

    Args:
        line_id: Metro line identifier (e.g., "amarela").
        start: Origin station name.
        end: Destination station name.

    Returns:
        Number of stations between start and end (inclusive of destination,
        exclusive of origin). Returns -1 if either station is not found.
    """
    stations = METRO_LINES.get(line_id, {}).get("stations", [])
    if not stations:
        return -1

    idx_start = _find_station_index(stations, start)
    idx_end = _find_station_index(stations, end)

    if idx_start < 0 or idx_end < 0:
        return -1

    return abs(idx_end - idx_start)


def _estimate_metro_time(station_count: int, transfers: int = 0) -> str:
    """
    Estimates travel time on the Lisbon Metro.

    Based on official Metro de Lisboa data:
    - ~2 minutes between consecutive stations (including stop time)
    - ~3 minutes for each line transfer (walking + waiting)
    - ~2 minutes average initial wait time

    Args:
        station_count: Number of stations to travel.
        transfers: Number of line transfers.

    Returns:
        Formatted time estimate string (e.g., "~12 min").
    """
    if station_count <= 0:
        return "~2 min"

    travel_min = station_count * 2  # 2 min per station
    transfer_min = transfers * 3    # 3 min per transfer
    wait_min = 2                    # Average initial wait
    total = travel_min + transfer_min + wait_min

    return f"~{total} min"


def _find_best_transfer_route(
    origin_lines: List[str],
    destination_lines: List[str],
    origin_station: str,
    destination_station: str,
) -> Optional[Dict[str, Any]]:
    """Finds the shortest valid metro transfer option between two stations."""
    transfer_stations = [
        ("Marquês de Pombal", ["amarela", "azul"]),
        ("Saldanha", ["amarela", "vermelha"]),
        ("Alameda", ["verde", "vermelha"]),
        ("Baixa-Chiado", ["azul", "verde"]),
        ("Campo Grande", ["amarela", "verde"]),
        ("São Sebastião", ["vermelha", "azul"]),
    ]

    candidates: List[Dict[str, Any]] = []
    for transfer_station, hub_lines in transfer_stations:
        origin_candidates = [line for line in origin_lines if line in hub_lines]
        destination_candidates = [line for line in destination_lines if line in hub_lines]

        for first_line in origin_candidates:
            for second_line in destination_candidates:
                if first_line == second_line:
                    continue

                leg1_count = _count_metro_stations(first_line, origin_station, transfer_station)
                leg2_count = _count_metro_stations(second_line, transfer_station, destination_station)
                if leg1_count < 0 or leg2_count < 0:
                    continue

                total_stations = leg1_count + leg2_count
                estimated_minutes = total_stations * 2 + 3 + 2
                candidates.append(
                    {
                        "station": transfer_station,
                        "first_line": first_line,
                        "second_line": second_line,
                        "leg1_count": leg1_count,
                        "leg2_count": leg2_count,
                        "total_stations": total_stations,
                        "estimated_minutes": estimated_minutes,
                    }
                )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item["estimated_minutes"],
            item["total_stations"],
            item["station"],
        )
    )
    return candidates[0]


def _get_line_status(line_id: str) -> str:
    """
    Gets real-time status for a specific Metro line.

    Args:
        line_id: Metro line identifier (e.g., "amarela").

    Returns:
        Status string ("ok" if normal, otherwise the disruption message).
        Returns "unknown" if status cannot be fetched.
    """
    try:
        metro_data = fetch_json_with_retry(METRO_STATUS_URL)
        if metro_data and metro_data.get('resposta'):
            return metro_data['resposta'].get(line_id, 'unknown').strip()
    except Exception:
        pass
    return "unknown"


def _build_route_source_line(sources: List[str]) -> str:
    """Builds a deduplicated source line for route answers."""
    deduped_sources: List[str] = []
    seen = set()
    for source in sources:
        if source and source not in seen:
            seen.add(source)
            deduped_sources.append(source)

    if not deduped_sources:
        return ""

    updated = datetime.now().strftime('%H:%M')
    return f"\n📌 **Fonte:** {' e '.join(deduped_sources)} **| Atualizado:** {updated}\n"


# Known place-name ambiguities for route planning. Maps a bare, normalized
# token to a localized preamble that asks the user to clarify what they meant
# before the tool renders a literal route to a Lisbon street of the same name.
_KNOWN_AMBIGUITIES: Dict[str, Dict[str, str]] = {
    "madeira": {
        "urban_name": "Rua Humberto Madeira",
        "alternate_name": "Ilha da Madeira",
        "alternate_hint": "A) 🏝️ **Ilha da Madeira** — não é acessível de metro; precisas de avião.",
        "urban_hint": "B) 🚇 **Rua Humberto Madeira / Av. Ilha da Madeira, em Lisboa** — sigo abaixo com a opção urbana mais próxima.",
    },
}


def _build_ambiguity_preamble(origin: str, destination: str) -> str:
    """Return a short ambiguity note when a bare ambiguous place token is used."""
    def _detect(value: str) -> Optional[Dict[str, str]]:
        token = (value or "").strip().lower()
        # Only flag a *bare* place name. "Rua Humberto Madeira" clearly means
        # the street, so we do not add the preamble there.
        if not token or " " in token:
            return None
        hit = _KNOWN_AMBIGUITIES.get(token)
        if not hit:
            return None
        return {**hit, "raw": value.strip()}

    hit = _detect(origin) or _detect(destination)
    if not hit:
        return ""

    name = hit["raw"].title()
    alternate_hint = hit["alternate_hint"]
    urban_hint = hit["urban_hint"]
    return (
        f"⚠️ **Ambiguidade no destino:** estás a perguntar sobre **{name}**?\n"
        f"- {alternate_hint}\n"
        f"- {urban_hint}\n"
        "- Assumo a interpretação urbana abaixo. Se não for isso, reformula o pedido com o nome completo do destino."
    )


# ==========================================================================
# LangChain Tools
# ==========================================================================

@tool
def get_route_between_stations(origin: str, destination: str) -> str:
    """
    Plans a multi-modal route between two locations using Metro, buses, and trains.

    This is the MAIN ROUTING TOOL for planning trips across Lisbon. It:
    - Detects Metro stations and shows direct/transfer routes
    - Identifies Lisbon landmarks (Colombo, Belém, etc.) and suggests best transport
    - Identifies CP train stations and suggests train connections
    - Recommends bus alternatives where appropriate

    For BUS-ONLY routes, use `find_bus_routes` instead.

    Args:
        origin: Starting location (Metro station, train station, or landmark).
        destination: Destination location.

    Returns:
        str: Multi-modal route suggestions with Metro, train, and bus options.
    """
    # Phase 1.4 ambiguity preamble: when a bare island/region name (currently
    # "Madeira") is used as origin or destination, Nominatim would silently
    # resolve it to "Rua Humberto Madeira" in Lisbon and we would render a
    # Metro route as if the user meant a street. That is misleading. Surface
    # the ambiguity explicitly so the user can disambiguate.
    ambiguity_note = _build_ambiguity_preamble(origin, destination)
    origin_display = _format_location_display_name(origin)
    destination_display = _format_location_display_name(destination)
    route_heading = f"🗺️ **Route: {origin_display} → {destination_display}**"
    response = f"{ambiguity_note}\n\n{route_heading}\n\n" if ambiguity_note else f"{route_heading}\n\n"
    sources_used: List[str] = []

    if _normalize_station(origin) == _normalize_station(destination):
        response += "✅ **You are already at the destination.**\n"
        response += "💡 No public transport leg is needed for this route.\n"
        return response + _build_route_source_line([])

    # Check if origin or destination is a known landmark
    origin_landmark = get_landmark_info(origin)
    dest_landmark = get_landmark_info(destination)

    # Check if both are Metro stations. Resolve common interchange aliases
    # before station-count and transfer calculations.
    metro_origin = _canonical_metro_station_name(origin)
    metro_destination = _canonical_metro_station_name(destination)
    origin_lines = get_station_lines(metro_origin)
    dest_lines = get_station_lines(metro_destination)

    # Check if they are CP train stations
    origin_cp = get_cp_station_info(origin)
    dest_cp = get_cp_station_info(destination)

    has_landmarks = bool(origin_landmark or dest_landmark)

    # Handle landmarks first
    if has_landmarks:
        response += "📍 **LOCATION INFORMATION**\n"

        if origin_landmark:
            response += f"**{_format_location_display_name(origin, detailed=True)}**\n"
            if origin_landmark.get('metro'):
                line = origin_landmark.get('line', '')
                line_emoji = METRO_LINES.get(line.split('/')[0], {}).get('emoji', '🚇')
                response += f"   🚇 Nearest Metro: **{origin_landmark['metro'].title()}** ({line_emoji} {line.title()} Line)\n"
            elif origin_landmark.get('alternative'):
                response += "   ⚠️ No direct Metro!\n"
                response += f"   🚌 Alternative: {origin_landmark['alternative']}\n"
            response += f"   ℹ️ {origin_landmark.get('description', '')}\n\n"

        if dest_landmark:
            response += f"**{_format_location_display_name(destination, detailed=True)}**\n"
            if dest_landmark.get('metro'):
                line = dest_landmark.get('line', '')
                line_emoji = METRO_LINES.get(line.split('/')[0], {}).get('emoji', '🚇')
                response += f"   🚇 Nearest Metro: **{dest_landmark['metro'].title()}** ({line_emoji} {line.title()} Line)\n"
            elif dest_landmark.get('alternative'):
                response += "   ⚠️ No direct Metro!\n"
                response += f"   🚌 Alternative: {dest_landmark['alternative']}\n"
            response += f"   ℹ️ {dest_landmark.get('description', '')}\n\n"

    # Resolve effective Metro stations (Handle Landmarks -> Stations)
    eff_origin = metro_origin if origin_lines else origin
    eff_dest = metro_destination if dest_lines else destination
    eff_origin_lines = origin_lines
    eff_dest_lines = dest_lines

    origin_from_landmark = False
    dest_from_landmark = False

    if origin_landmark and origin_landmark.get('metro'):
        eff_origin = origin_landmark['metro']
        eff_origin_lines = get_station_lines(eff_origin)
        origin_from_landmark = True

    if dest_landmark and dest_landmark.get('metro'):
        eff_dest = dest_landmark['metro']
        eff_dest_lines = get_station_lines(eff_dest)
        dest_from_landmark = True

    eff_origin_cp = origin_cp
    eff_dest_cp = dest_cp
    origin_train_station = origin.title()
    dest_train_station = destination.title()

    if not eff_origin_cp and origin_landmark and origin_landmark.get("train_station"):
        origin_train_station = str(origin_landmark["train_station"]).strip()
        eff_origin_cp = get_cp_station_info(origin_train_station)

    if not eff_dest_cp and dest_landmark and dest_landmark.get("train_station"):
        dest_train_station = str(dest_landmark["train_station"]).strip()
        eff_dest_cp = get_cp_station_info(dest_train_station)

    # Calculate Metro Route
    if eff_origin_lines and eff_dest_lines:
        sources_used.append("[*Metro de Lisboa*](https://www.metrolisboa.pt)")
        response += "🚇 **METRO ROUTE**\n"

        common_lines = set(eff_origin_lines) & set(eff_dest_lines)

        if common_lines:
            response += "✅ **Direct Route Available**\n\n"
            for line in common_lines:
                line_info = METRO_LINES.get(line, {})
                emoji = line_info.get('emoji', '')
                name = line_info.get('name', line.title())
                direction = _get_metro_direction(line, eff_origin, eff_dest)

                # B1: Check real-time line status
                line_status = _get_line_status(line)
                if line_status.lower() not in ('ok', 'unknown', ''):
                    response += f"   ⚠️ **Line Alert**: {line_status}\n"

                # B4: Travel time estimate
                station_count = _count_metro_stations(line, eff_origin, eff_dest)
                time_est = _estimate_metro_time(station_count) if station_count > 0 else ""
                stations_str = f" ({station_count} stations)" if station_count > 0 else ""

                response += f"   {emoji} Take **{line.title()} Line** ({name})\n"
                if time_est:
                    response += f"   ⏱️ Estimated travel time: **{time_est}**{stations_str}\n"

                step = 1
                if origin_from_landmark:
                    response += f"   {step}. Walk from {origin_display} to **{eff_origin.title()}**\n"
                    step += 1

                response += f"   {step}. Board at **{eff_origin.title()}** {direction}\n"
                step += 1

                response += f"   {step}. Exit at **{eff_dest.title()}**\n"

                if dest_from_landmark:
                    step += 1
                    response += f"   {step}. Walk to {destination_display}\n"

                response += "\n"

        else:
            response += "🔄 **Transfer Required**\n\n"
            best_transfer = _find_best_transfer_route(
                origin_lines=eff_origin_lines,
                destination_lines=eff_dest_lines,
                origin_station=eff_origin,
                destination_station=eff_dest,
            )

            if best_transfer:
                best_hub = best_transfer["station"]
                l1 = best_transfer["first_line"]
                l2 = best_transfer["second_line"]
                l1_info = METRO_LINES[l1]
                l2_info = METRO_LINES[l2]

                # B1: Check real-time status for both lines
                for check_line, check_info in [(l1, l1_info), (l2, l2_info)]:
                    status = _get_line_status(check_line)
                    if status.lower() not in ('ok', 'unknown', ''):
                        response += f"   ⚠️ **{check_info['emoji']} {check_line.title()} Line Alert**: {status}\n"

                # B4: Total travel time (leg 1 + transfer + leg 2)
                total_stations = best_transfer["total_stations"]
                time_est = _estimate_metro_time(total_stations, transfers=1)

                response += f"   💡 **Transfer at**: {best_hub} ({l1_info['emoji']} ↔ {l2_info['emoji']})\n"
                response += f"   ⏱️ Estimated travel time: **{time_est}** ({total_stations} stations + 1 transfer)\n\n"
                response += "   **Full Route**:\n"

                step = 1
                if origin_from_landmark:
                    response += f"   {step}. Walk from {origin_display} to **{eff_origin.title()}**\n"
                    step += 1

                dir1 = _get_metro_direction(l1, eff_origin, best_hub)
                response += f"   {step}. {l1_info['emoji']} Board at **{eff_origin.title()}** {dir1}\n"
                step += 1
                response += f"   {step}. Exit at **{best_hub}**\n"
                step += 1

                dir2 = _get_metro_direction(l2, best_hub, eff_dest)
                response += f"   {step}. {l2_info['emoji']} Transfer to **{l2_info['name']}** {dir2}\n"
                step += 1
                response += f"   {step}. Exit at **{eff_dest.title()}**\n"

                if dest_from_landmark:
                    step += 1
                    response += f"   {step}. Walk to {destination_display}\n"

                response += "\n"
            else:
                response += "⚠️ Route requires complex transfer. Check [Metro map](https://www.metrolisboa.pt/viajar/mapas-e-diagramas/).\n\n"

    elif eff_origin_lines:
        # Origin valid, Dest invalid
        response += f"🚇 **Origin is Metro**: {eff_origin.title()}\n"
        if origin_from_landmark:
            response += f"   (Nearest station to {origin})\n"
        response += f"❌ Destination '{destination_display}' not on Metro.\n"
        response += "   Consider using Carris buses or CP trains.\n\n"

    elif eff_dest_lines:
        # Dest valid, Origin invalid
        response += f"❌ Origin '{origin_display}' not on Metro.\n"
        response += f"🚇 **Destination is Metro**: {eff_dest.title()}\n"
        if dest_from_landmark:
            response += f"   (Nearest station to {destination})\n"
        response += "   Consider using Carris buses or CP trains to reach the Metro.\n\n"

    else:
        if not has_landmarks:  # Only print if we haven't printed landmark info
            response += "❌ Neither location is a known Metro station.\n\n"

    # Check for CP Train options (only when BOTH ends are CP stations)
    # If only one end is a CP station, the metro route above is sufficient.
    if eff_origin_cp and eff_dest_cp:
        response += "🚆 **CP TRAINS**\n"
        sources_used.append("[*CP*](https://www.cp.pt)")

        common_lines = set(eff_origin_cp.get("lines", [])) & set(eff_dest_cp.get("lines", []))

        if common_lines:
            response += "✅ **Direct Train Route Available**\n\n"
            for line in common_lines:
                line_info = CP_LINES.get(line, {"name": line.title()})
                response += f"   🚆 Take **{line_info['name']}**\n"
                step = 1
                if origin_landmark and origin_train_station.lower() != origin.lower():
                    response += f"   {step}. Walk from {origin_display} to **{origin_train_station}**\n"
                    step += 1
                response += f"   {step}. 📍 Board at: **{origin_train_station}**\n"
                step += 1
                response += f"   {step}. 📍 Exit at: **{dest_train_station}**\n"
                if dest_landmark and dest_train_station.lower() != destination.lower():
                    step += 1
                    response += f"   {step}. Walk to {destination_display}\n"
                if line_info.get("frequency"):
                    response += f"   🕒 Frequency: {line_info['frequency']}\n"
                response += "\n"
            return response + _build_route_source_line(sources_used)
        else:
            response += f"⚠️ No direct train line linking {origin_train_station} and {dest_train_station}.\n"
            response += "   You may need to transfer at a major hub (e.g., Entrecampos, Oriente, Sete Rios).\n\n"

    return response + _build_route_source_line(sources_used)


@tool
def get_transport_summary(language: str = "pt") -> str:
    """
    Gets a quick summary of all public transport status in Lisbon.
    Combines Metro, buses, and trains into a single overview.

    Args:
        language: Preferred response language. Use ``"pt"`` for Portuguese
            and ``"en"`` for English. Defaults to Portuguese for backward
            compatibility with existing tool calls.

    Returns:
        str: Combined transport status summary.
    """
    is_pt = (language or "pt").lower().startswith("pt")
    now_str = datetime.now().strftime('%H:%M')
    title = "Ponto de situação dos transportes em Lisboa" if is_pt else "Transport Status in Lisbon"
    direct_answer = (
        f"Resumo rápido do Metro, autocarros e comboios atualizado às **{now_str}**."
        if is_pt
        else f"Quick status for Metro, buses, and suburban trains updated at **{now_str}**."
    )
    source_label = "Fonte" if is_pt else "Source"
    updated_label = "Atualizado" if is_pt else "Updated"
    response = "\n".join(
        [
            f"### 🔵 **{title}**",
            "",
            f"✅ **{'Resposta direta' if is_pt else 'Direct answer'}:** {direct_answer}",
            "",
        ]
    )

    # 1. Metro Status
    response += "- **🚇 Metro de Lisboa**\n"

    metro_data = fetch_json_with_retry(METRO_STATUS_URL)
    if metro_data and metro_data.get('resposta'):
        resp = metro_data['resposta']
        all_ok = True
        line_labels = {
            "amarela": "Amarela" if is_pt else "Yellow",
            "azul": "Azul" if is_pt else "Blue",
            "verde": "Verde" if is_pt else "Green",
            "vermelha": "Vermelha" if is_pt else "Red",
        }
        for line_key, line_info in METRO_LINES.items():
            status = str(resp.get(line_key, "Unknown")).strip() or "Unknown"
            if status.lower() != 'ok':
                all_ok = False
            status_label = "Ok" if status.lower() == "ok" else status
            response += f"    - {line_info['emoji']} **{line_labels.get(line_key, line_key.title())}:** {status_label}\n"

        if all_ok:
            status_text = "Circulação normal em todas as linhas" if is_pt else "Normal service on all lines"
            response += f"    - ✅ **{'Estado geral' if is_pt else 'Overall status'}:** {status_text}\n"
        else:
            warning_text = (
                "Há perturbações reportadas; confirma a linha afetada antes de sair"
                if is_pt
                else "Disruptions are reported; check the affected line before leaving"
            )
            response += f"    - ⚠️ **{'Estado geral' if is_pt else 'Overall status'}:** {warning_text}\n"
    else:
        response += f"    - ❌ **{'Estado' if is_pt else 'Status'}:** {'Dados indisponíveis' if is_pt else 'Data unavailable'}\n"

    response += "\n"

    # 2. Carris (Urban Lisbon)
    response += "- **🚌 Carris Urban**\n"

    try:
        from tools.carris_api import fetch_gtfs_rt_vehicles

        vehicles = fetch_gtfs_rt_vehicles()
        if vehicles:
            metric = "Veículos em serviço" if is_pt else "Vehicles in service"
            if is_pt:
                vehicle_word = "veículo" if len(vehicles) == 1 else "veículos"
            else:
                vehicle_word = "vehicle" if len(vehicles) == 1 else "vehicles"
            response += f"    - ✅ **{metric}:** {len(vehicles)} {vehicle_word}\n"
        else:
            status_text = "Dados em tempo real indisponíveis" if is_pt else "Real-time data unavailable"
            response += f"    - ⚠️ **{'Estado' if is_pt else 'Status'}:** {status_text}\n"
    except Exception as e:
        logger.warning(f"Carris Urban data failed: {e}")
        response += f"    - ⚠️ **{'Estado' if is_pt else 'Status'}:** {'Dados indisponíveis' if is_pt else 'Data unavailable'}\n"

    response += "\n"

    # 3. Carris Metropolitana (AML metropolitan buses)
    response += "- **🚌 Carris Metropolitana**\n"

    try:
        from tools.carrismetropolitana_api import (
            CARRIS_ALERTS_URL,
        )

        alerts_data = fetch_json_with_retry(CARRIS_ALERTS_URL)
        if alerts_data:
            # API returns a list directly, not a dict with 'entity' key.
            alerts = alerts_data if isinstance(alerts_data, list) else alerts_data.get('entity', [])
            if alerts:
                metric = "Alertas ativos" if is_pt else "Active alerts"
                if is_pt:
                    alert_word = "alerta" if len(alerts) == 1 else "alertas"
                else:
                    alert_word = "alert" if len(alerts) == 1 else "alerts"
                response += f"    - ⚠️ **{metric}:** {len(alerts)} {alert_word}\n"
            else:
                response += f"    - ✅ **{'Estado' if is_pt else 'Status'}:** {'Sem alertas ativos' if is_pt else 'No active alerts'}\n"
        else:
            status_text = "Dados de alertas indisponíveis" if is_pt else "Alert data unavailable"
            response += f"    - ⚠️ **{'Estado' if is_pt else 'Status'}:** {status_text}\n"
    except Exception as e:
        logger.warning(f"Carris Metropolitana alerts failed: {e}")
        response += f"    - ⚠️ **{'Estado' if is_pt else 'Status'}:** {'Dados indisponíveis' if is_pt else 'Data unavailable'}\n"

    response += "\n"

    # 4. CP suburban trains
    cp_title = "Comboios suburbanos CP em Lisboa/AML" if is_pt else "CP Suburban Trains in Lisbon/AML"
    response += f"- **🚆 {cp_title}**\n"

    try:
        aml_trains = get_cp_aml_trains()
        if aml_trains:
            total = len(aml_trains)
            delayed = sum(1 for t in aml_trains if (t.get('delay') or 0) > 60)

            trains_metric = "Comboios a circular na AML" if is_pt else "Trains currently in the AML"
            if is_pt:
                train_word = "comboio" if total == 1 else "comboios"
            else:
                train_word = "train" if total == 1 else "trains"
            response += f"    - 📊 **{trains_metric}:** {total} {train_word}\n"
            if delayed > 0:
                delay_metric = "Atrasos superiores a 1 min" if is_pt else "Delays over 1 min"
                response += f"    - ⚠️ **{delay_metric}:** {delayed} {train_word}\n"
            else:
                status_text = "Comboios a operar normalmente" if is_pt else "Trains operating normally"
                response += f"    - ✅ **{'Estado' if is_pt else 'Status'}:** {status_text}\n"
        else:
            response += f"    - ⚠️ **{'Estado' if is_pt else 'Status'}:** {'Dados indisponíveis' if is_pt else 'Data unavailable'}\n"
    except Exception as e:
        logger.warning(f"CP train data failed: {e}")
        response += f"    - ⚠️ **{'Estado' if is_pt else 'Status'}:** {'Dados indisponíveis' if is_pt else 'Data unavailable'}\n"

    response += "\n"
    if is_pt:
        response += (
            "💡 **Antes de sair:**\n"
            "- Se vais usar Carris Metropolitana ou CP, confirma a partida específica pouco antes de sair, porque alertas e atrasos agregados não identificam sempre a tua linha.\n\n"
        )
    else:
        response += (
            "💡 **Before leaving:**\n"
            "- If you plan to use Carris Metropolitana or CP, check the specific departure shortly before you leave because aggregate alerts and delays do not always identify your line.\n\n"
        )

    response += f"📌 **{source_label}:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | [*Carris*](https://www.carris.pt) | [*Carris Metropolitana*](https://www.carrismetropolitana.pt) | [*CP*](https://www.cp.pt) | **{updated_label}:** {now_str}\n"

    return response


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 70 + "\033[0m")
    print("\033[1m\U0001f9ea MULTI-MODAL TRANSPORT API - COMPREHENSIVE TEST SUITE\033[0m")
    print("\033[1m" + "=" * 70 + "\033[0m")

    test_results = {"passed": 0, "failed": 0, "total": 0}

    def run_test(name, func, args=None):
        """Runs a test and tracks results."""
        test_results["total"] += 1
        print(f"\n{'=' * 60}")
        print(f"\033[1m\U0001f9ea TEST {test_results['total']}: {name}\033[0m")
        print(f"{'=' * 60}")
        try:
            result = func(args if args else {})
            if result:
                test_results["passed"] += 1
                print(f"\033[1;32m[PASS]\033[0m Result length: {len(result)} chars")
                # # Show first 600 chars for readability
                # print(result[:600])
                # if len(result) > 600:
                #     print(f"... ({len(result) - 600} more chars)")
                print(result)
            else:
                test_results["failed"] += 1
                print("\033[1;31m[FAIL]\033[0m Empty result")
        except Exception as e:
            test_results["failed"] += 1
            print(f"\033[1;31m[FAIL]\033[0m Error: {e}")

    # =========================================================================
    # HELPER FUNCTION TESTS
    # =========================================================================

    # TEST: Station counting (internal validation)
    print(f"\n{'=' * 60}")
    print("\033[1m\U0001f9ea INTERNAL: _count_metro_stations validation\033[0m")
    print(f"{'=' * 60}")

    # Helper variables for colors to avoid f-string backslash errors in <= 3.11
    OK_TXT = "\033[32mOK\033[0m"
    FAIL_TXT = "\033[31mFAIL\033[0m"
    FAIL_12_TXT = "\033[31mFAIL (expected 12)\033[0m"
    FAIL_2_TXT = "\033[31mFAIL (expected 2)\033[0m"

    # Amarela: rato(0) to odivelas(12) = 12 stations
    count = _count_metro_stations("amarela", "rato", "odivelas")
    print(f"  Rato -> Odivelas (Amarela): {count} stations {OK_TXT if count == 12 else FAIL_12_TXT}")

    # Verde: cais do sodre(0) to telheiras(12) = 12 stations
    count = _count_metro_stations("verde", "cais do sodre", "telheiras")
    print(f"  Cais do Sodre -> Telheiras (Verde): {count} stations {OK_TXT if count == 12 else FAIL_12_TXT}")

    # Azul: santa apolonia(0) to baixa-chiado(2) = 2 stations
    count = _count_metro_stations("azul", "santa apolonia", "baixa-chiado")
    print(f"  Santa Apolonia -> Baixa-Chiado (Azul): {count} stations {OK_TXT if count == 2 else FAIL_2_TXT}")

    # Time estimation
    time_est = _estimate_metro_time(5, transfers=0)
    print(f"  Time estimate (5 stations, 0 transfers): {time_est} {OK_TXT if '12' in time_est else FAIL_TXT}")

    time_est = _estimate_metro_time(8, transfers=1)
    print(f"  Time estimate (8 stations, 1 transfer): {time_est} {OK_TXT if '21' in time_est else FAIL_TXT}")

    # =========================================================================
    # METRO ROUTE TESTS - Direct Routes
    # =========================================================================

    # TEST 1: Direct route on same line (Vermelha)
    run_test(
        "Direct Metro Route - Same Line (Aeroporto -> Saldanha) [VERMELHA]",
        get_route_between_stations.invoke,
        {"origin": "Aeroporto", "destination": "Saldanha"}
    )

    # TEST 2: Direct route on same line (Verde)
    run_test(
        "Direct Metro Route - Same Line (Cais do Sodre -> Arroios) [VERDE]",
        get_route_between_stations.invoke,
        {"origin": "Cais do Sodré", "destination": "Arroios"}
    )

    # =========================================================================
    # METRO ROUTE TESTS - Transfer Required
    # =========================================================================

    # TEST 3: Transfer route (Azul -> Vermelha via Sao Sebastiao)
    run_test(
        "Transfer Route - Reboleira -> Aeroporto [AZUL -> VERMELHA]",
        get_route_between_stations.invoke,
        {"origin": "Reboleira", "destination": "Aeroporto"}
    )

    # TEST 4: Transfer route (Amarela -> Verde via Campo Grande)
    run_test(
        "Transfer Route - Odivelas -> Rossio [AMARELA -> VERDE]",
        get_route_between_stations.invoke,
        {"origin": "Odivelas", "destination": "Rossio"}
    )

    # =========================================================================
    # LANDMARK ROUTING TESTS
    # =========================================================================

    # TEST 5: Landmark routing (Colombo -> Oriente)
    run_test(
        "Landmark Route - Colombo -> Oriente [LANDMARK + METRO]",
        get_route_between_stations.invoke,
        {"origin": "Colombo", "destination": "Oriente"}
    )

    # TEST 6: Landmark with no metro (Belem)
    run_test(
        "Landmark Route - Belem (no Metro) [ALTERNATIVE TRANSPORT]",
        get_route_between_stations.invoke,
        {"origin": "Aeroporto", "destination": "Belém"}
    )

    # =========================================================================
    # EDGE CASES
    # =========================================================================

    # TEST 7: Unknown locations
    run_test(
        "Edge Case - Unknown Origin and Destination",
        get_route_between_stations.invoke,
        {"origin": "Praia do Guincho", "destination": "Serra da Estrela"}
    )

    # TEST 8: Same station
    run_test(
        "Edge Case - Same Origin and Destination",
        get_route_between_stations.invoke,
        {"origin": "Saldanha", "destination": "Saldanha"}
    )

    # =========================================================================
    # TRANSPORT SUMMARY TEST
    # =========================================================================

    # TEST 9: Full transport summary
    run_test(
        "Transport Summary - All Modes [METRO + CARRIS + CP]",
        get_transport_summary.invoke
    )

    # TEST 10: Same-origin short circuit
    run_test(
        "Regression - Same Location Short-Circuit (Saldanha -> Saldanha)",
        get_route_between_stations.invoke,
        {"origin": "Saldanha", "destination": "Saldanha"}
    )

    # TEST 11: CP direct route preservation
    run_test(
        "Regression - CP Direct Route (Rossio -> Sintra)",
        get_route_between_stations.invoke,
        {"origin": "Rossio", "destination": "Sintra"}
    )

    # TEST 12: Metro direct route used by transport-agent fast path
    run_test(
        "Regression - Metro Route (Saldanha -> Odivelas)",
        get_route_between_stations.invoke,
        {"origin": "Saldanha", "destination": "Odivelas"}
    )

    # =========================================================================
    # TEST SUMMARY
    # =========================================================================

    print("\n" + "=" * 70)
    print("\033[1m\U0001f4ca TEST SUMMARY\033[0m")
    print("=" * 70)
    print(f"\033[1;32m\u2705 Passed: {test_results['passed']}/{test_results['total']}\033[0m")
    print(f"\033[1;31m\u274c Failed: {test_results['failed']}/{test_results['total']}\033[0m")

    if test_results['failed'] == 0:
        print("\n\033[1;32m🎉 ALL TESTS PASSED! Transport system is working correctly.\033[0m")
    else:
        print("\n\033[1;33m⚠️  Some tests failed. Check errors above.\033[0m")

    print("=" * 70 + "\n")
