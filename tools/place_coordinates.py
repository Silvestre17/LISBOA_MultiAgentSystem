# ==========================================================================
# Master Thesis - VisitLisboa Place Coordinates
#   - André Filipe Gomes Silvestre, 20240502
#
#   Coordinates for the VisitLisboa place catalogue, which ships addresses
#   but no coordinates. Proximity ranking ("viewpoints near Alfama") used to
#   fall back to postal-code centroids, so places whose address has no postal
#   code (most miradouros) could not be ranked and sank below rooftop bars.
#   This module geocodes each place once, stores the result next to
#   places.json, and serves it at query time without any network call.
#
#   Usage:
#       > python -m tools.place_coordinates
#           Geocode the places that have no stored coordinates yet.
#       > python -m tools.place_coordinates --refresh
#           Geocode every place again.
#       > python -m tools.place_coordinates --refresh-landmarks
#           Geocode the landmarks (museums, monuments, viewpoints...) again.
# ==========================================================================

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from config import Config
except ModuleNotFoundError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import Config

logger = logging.getLogger(__name__)

PLACE_COORDINATES_PATH = Path(Config.PATH_VISIT_LISBOA_PLACES).with_name("places_coordinates.json")
# Accommodation is left out: itineraries and proximity searches do not rank hotels.
_SKIPPED_CATEGORIES = {
    "hotel",
    "apartments & hotel apartments",
    "local & rural accommodation",
    "guest houses",
    "hostels",
    "pousadas",
    "camping",
}


@lru_cache(maxsize=1)
def load_place_coordinates() -> Dict[str, Tuple[float, float]]:
    """Return stored coordinates keyed by VisitLisboa place URL."""
    try:
        payload = json.loads(PLACE_COORDINATES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    coordinates: Dict[str, Tuple[float, float]] = {}
    for url, entry in (payload.get("places") or {}).items():
        try:
            coordinates[str(url)] = (float(entry["lat"]), float(entry["lon"]))
        except (KeyError, TypeError, ValueError):
            continue
    return coordinates


def lookup_place_coordinates(url: Optional[str]) -> Optional[Tuple[float, float]]:
    """Return stored coordinates for a place URL, if known."""
    if not url:
        return None
    return load_place_coordinates().get(str(url).strip())


def _significant_tokens(text: str) -> set[str]:
    """Return lower-case name tokens that can confirm a geocoder match."""
    from tools.location_resolver import normalize_location_text

    generic = {"lisboa", "lisbon", "portugal", "rua", "avenida", "av", "largo", "praca", "travessa", "calcada", "the", "and", "de", "da", "do", "dos", "das"}
    return {
        token for token in normalize_location_text(text).split()
        if len(token) >= 4 and not token.isdigit() and token not in generic
    }


# Lisbon municipality, with a margin that keeps Parque das Nações and Belém.
_LISBON_CITY_BOX = (38.68, 38.80, -9.24, -9.08)
_LANDMARK_CATEGORIES = {
    "view points",
    "museums",
    "monuments",
    "museums & monuments",
    "attractions",
    "nature",
    "only in lisbon",
    "cultural centres",
    "cultural centres & trips",
    "family & kids",
    "parks & gardens",
    "beaches",
}


def _in_lisbon_city(lat: float, lon: float) -> bool:
    """Return whether a point lies in the Lisbon municipality bounding box."""
    lat_min, lat_max, lon_min, lon_max = _LISBON_CITY_BOX
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


_MIN_REQUEST_INTERVAL_S = 1.1
_last_request_at = 0.0


def _photon_search(query: str, attempts: int = 4) -> List[Dict[str, Any]]:
    """Query Photon politely: at most one request per second, retrying failures.

    The public Photon instance throttles bursts, so the build is sequential,
    spaced, and retries a failed request with a growing pause instead of
    recording the place as missing.
    """
    global _last_request_at
    from tools.location_resolver import _GeocoderUnavailable, _fetch_photon_results_cached

    for attempt in range(attempts):
        wait = _MIN_REQUEST_INTERVAL_S - (time.time() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.time()
        try:
            return list(_fetch_photon_results_cached(query))
        except _GeocoderUnavailable as exc:
            logger.info("%s (attempt %d)", exc, attempt + 1)
            time.sleep(2.0 * (attempt + 1))
    return []


def _wikipedia_landmark(title: str, in_lisbon: bool) -> Optional[Dict[str, Any]]:
    """Return a landmark's coordinates from its Wikipedia page (or Wikidata item).

    A street address places a landmark anywhere on a long avenue ("Avenida
    de Brasília" put the Monument to the Discoveries at the Champalimaud
    Centre); the encyclopaedia page gives the monument itself.
    """
    from tools.web_knowledge import wikipedia_place_card

    card = wikipedia_place_card(title, "en")
    if not card:
        return None
    lat, lon = float(card["lat"]), float(card["lon"])
    if in_lisbon and not _in_lisbon_city(lat, lon):
        return None
    return {"lat": round(lat, 6), "lon": round(lon, 6), "source": "wikipedia"}


def _geocode_place(place: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Geocode one catalogue place: by encyclopaedia page, by name, or by street address.

    Landmarks are looked up on Wikipedia first. Photon (OpenStreetMap) is then
    queried directly, once per candidate query. A name match must share the
    place's distinctive words; an address match must share the street words
    and, when both sides show one, the postal prefix. Places whose address
    ends in "Lisboa" must fall in the city.
    """
    import re

    title = str(place.get("title") or "").strip()
    address = str(place.get("location") or "").strip()
    in_lisbon = bool(re.search(r"\blisboa\b\s*$", address.lower())) or not address
    postal = re.search(r"\b(\d{4})\s*-\s*\d{3}\b", address)
    title_query = ("title", f"{title}, Lisboa" if in_lisbon else title)
    address_query = ("address", address)
    # Landmarks are best found by name; businesses (often chains with several
    # branches) by their street address.
    landmark = str(place.get("category") or "").strip().lower() in _LANDMARK_CATEGORIES
    if landmark and title:
        found = _wikipedia_landmark(title, in_lisbon)
        if found:
            return found
    queries = [title_query, address_query] if landmark else [address_query, title_query]
    for source, query in queries:
        if not query or query.strip(" ,.").lower() in {"lisboa", "lisbon"}:
            continue
        expected = _significant_tokens(title if source == "title" else address.split(",")[0])
        for result in _photon_search(query):
            lat, lon = float(result["lat"]), float(result["lon"])
            if in_lisbon and not _in_lisbon_city(lat, lon):
                continue
            display = str(result.get("display_name") or "")
            overlap = expected & _significant_tokens(display)
            if expected and len(overlap) < min(2, len(expected)):
                continue
            result_postal = re.search(r"\b(\d{4})-\d{3}\b", display)
            if postal and result_postal and postal.group(1) != result_postal.group(1):
                continue
            return {"lat": round(lat, 6), "lon": round(lon, 6), "source": source}
    return None


def build_place_coordinates(refresh: bool = False, refresh_landmarks: bool = False) -> Dict[str, Any]:
    """Geocode catalogue places and save the coordinate file.

    The file is saved every 50 places, so an interrupted build keeps its
    progress and the next run continues from the missing places.

    Args:
        refresh: Geocode every place again instead of only the missing ones.
        refresh_landmarks: Geocode the landmark categories again; a landmark
            that is not found again keeps its stored coordinates.

    Returns:
        The saved payload.
    """
    places: List[Dict[str, Any]] = json.loads(Path(Config.PATH_VISIT_LISBOA_PLACES).read_text(encoding="utf-8"))
    try:
        existing = json.loads(PLACE_COORDINATES_PATH.read_text(encoding="utf-8")).get("places") or {}
    except (OSError, ValueError):
        existing = {}

    stored: Dict[str, Any] = {} if refresh else dict(existing)
    catalogue_urls = {str(place.get("url") or "").strip() for place in places}
    stored = {url: entry for url, entry in stored.items() if url in catalogue_urls}
    if refresh_landmarks:
        from tools.utils import haversine_distance

        for place in places:
            url = str(place.get("url") or "").strip()
            if url in stored and str(place.get("category") or "").strip().lower() in _LANDMARK_CATEGORIES:
                found = _wikipedia_landmark(str(place.get("title") or "").strip(), _address_in_lisbon(place))
                current = stored[url]
                # The page point replaces a geocoded one nearby (a street
                # address placed the monument elsewhere on the avenue); a far
                # or minute-rounded page point (a park's centroid) does not.
                if found and not _minute_rounded(found) and haversine_distance(
                    float(current["lat"]), float(current["lon"]), found["lat"], found["lon"]
                ) <= 3.0:
                    stored[url] = {"title": place.get("title"), **found}
    pending = [
        place for place in places
        if str(place.get("url") or "").strip()
        and str(place.get("url") or "").strip() not in stored
        and str(place.get("category") or "").strip().lower() not in _SKIPPED_CATEGORIES
    ]
    print(f"Geocoding {len(pending)} places ({len(stored)} already stored)...", flush=True)
    payload: Dict[str, Any] = {}
    for index, place in enumerate(pending, start=1):
        entry = _geocode_place(place)
        if entry:
            stored[str(place["url"]).strip()] = {"title": place.get("title"), **entry}
        if index % 50 == 0 or index == len(pending):
            payload = _save_coordinates(stored)
            print(f"   {index}/{len(pending)} done ({len(stored)} stored)", flush=True)
    if not pending:
        payload = _save_coordinates(stored)
    print(f"Saved {len(stored)} coordinates to {PLACE_COORDINATES_PATH}", flush=True)
    return payload


def _minute_rounded(point: Dict[str, Any]) -> bool:
    """Return whether coordinates are only given to the minute (about 1 km)."""
    return all(abs(float(point[key]) * 60 - round(float(point[key]) * 60)) < 1e-3 for key in ("lat", "lon"))


def _address_in_lisbon(place: Dict[str, Any]) -> bool:
    """Return whether a catalogue address ends in "Lisboa" (or is missing)."""
    import re

    address = str(place.get("location") or "").strip()
    return bool(re.search(r"\blisboa\b\s*$", address.lower())) or not address


def _save_coordinates(stored: Dict[str, Any]) -> Dict[str, Any]:
    """Write the coordinate file atomically and refresh the in-memory copy."""
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "VisitLisboa places geocoded with OpenStreetMap (Photon); landmarks from Wikipedia/Wikidata",
        "places": dict(sorted(stored.items())),
    }
    temp_path = PLACE_COORDINATES_PATH.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temp_path, PLACE_COORDINATES_PATH)
    load_place_coordinates.cache_clear()
    return payload


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Geocode the VisitLisboa place catalogue.")
    parser.add_argument("--refresh", action="store_true", help="Geocode every place again.")
    parser.add_argument("--refresh-landmarks", action="store_true", help="Geocode the landmarks again from Wikipedia.")
    args = parser.parse_args()
    build_place_coordinates(refresh=args.refresh, refresh_landmarks=args.refresh_landmarks)


if __name__ == "__main__":
    main()
