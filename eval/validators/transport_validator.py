# ==========================================================================
# Master Thesis - Transport Route Validator
#   - André Filipe Gomes Silvestre, 20240502
#
# Deterministic validators for Lisbon Metro route facts.
# Uses the canonical METRO_LINES and METRO_STATIONS data from
# tools/metrolisboa_api.py as ground truth.
#
# These validators complement the LLM-as-a-Judge by providing
# binary (pass/fail) checks that do not depend on LLM judgment.
# ==========================================================================

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Import canonical metro data
from tools.metrolisboa_api import METRO_STATIONS

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _normalize(name: str) -> str:
    """Normalize a station name for comparison.

    Strips accents, lowercases, removes hyphens, and collapses whitespace.

    Args:
        name: Raw station name string.

    Returns:
        Normalized string for fuzzy matching.
    """
    name = name.lower().strip()
    # Remove accents
    nfkd = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Normalize separators
    name = name.replace("-", " ").replace("/", " ")
    return re.sub(r"\s+", " ", name).strip()


def _find_station(name: str) -> str | None:
    """Resolve a station name to its canonical key in METRO_STATIONS.

    Args:
        name: Station name (possibly with accents or typos).

    Returns:
        Canonical key if found, else None.
    """
    norm = _normalize(name)
    # Direct lookup (normalized keys)
    for key in METRO_STATIONS:
        if _normalize(key) == norm:
            return key
    # Substring fallback
    for key in METRO_STATIONS:
        if norm in _normalize(key) or _normalize(key) in norm:
            return key
    return None


def _get_station_lines(station_key: str) -> list[str]:
    """Return the list of metro lines serving a station.

    Args:
        station_key: Canonical station key from METRO_STATIONS.

    Returns:
        List of line names (e.g. ["amarela", "azul"]).
    """
    return METRO_STATIONS.get(station_key, [])


# -----------------------------------------------------------------------
# Transfer Hub Knowledge
# -----------------------------------------------------------------------

TRANSFER_HUBS: dict[str, list[str]] = {
    "marques de pombal": ["amarela", "azul"],
    "saldanha": ["amarela", "vermelha"],
    "campo grande": ["amarela", "verde"],
    "alameda": ["verde", "vermelha"],
    "sao sebastiao": ["azul", "vermelha"],
    "baixa-chiado": ["azul", "verde"],
}


# -----------------------------------------------------------------------
# Public API: Validators
# -----------------------------------------------------------------------

def validate_station_exists(station_name: str) -> dict[str, Any]:
    """Check whether a station name resolves to a real Metro station.

    Args:
        station_name: Station to validate.

    Returns:
        Dict with keys: valid (bool), canonical_name (str|None),
        lines (list[str]).
    """
    key = _find_station(station_name)
    if key is None:
        return {"valid": False, "canonical_name": None, "lines": []}
    return {"valid": True, "canonical_name": key, "lines": _get_station_lines(key)}


def validate_station_on_line(station_name: str, line: str) -> bool:
    """Check whether a station is actually on the specified line.

    Args:
        station_name: Station to check.
        line: Metro line name (e.g. "verde").

    Returns:
        True if station exists on that line.
    """
    key = _find_station(station_name)
    if key is None:
        return False
    return line.lower() in _get_station_lines(key)


def validate_transfer_point(station_name: str, line_a: str, line_b: str) -> dict[str, Any]:
    """Check whether a station is a valid transfer between two lines.

    Args:
        station_name: Transfer station.
        line_a: First line.
        line_b: Second line.

    Returns:
        Dict with keys: valid (bool), station (str|None),
        actual_lines (list[str]).
    """
    key = _find_station(station_name)
    if key is None:
        return {"valid": False, "station": None, "actual_lines": []}
    lines = _get_station_lines(key)
    la, lb = line_a.lower(), line_b.lower()
    ok = la in lines and lb in lines
    return {"valid": ok, "station": key, "actual_lines": lines}


def validate_metro_route(
    origin: str,
    destination: str,
    claimed_line: str | None = None,
    claimed_transfer: str | None = None,
    claimed_transfer_lines: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Validate a metro route claim against ground truth.

    This checks:
    1. Both stations exist.
    2. If claimed_line is given, both stations are on it (direct route).
    3. If a transfer is claimed, the transfer station connects the two lines.

    Args:
        origin: Origin station name.
        destination: Destination station name.
        claimed_line: Line claimed for direct travel (optional).
        claimed_transfer: Transfer station claimed (optional).
        claimed_transfer_lines: Tuple of (line_from, line_to) at transfer.

    Returns:
        Dict with route_valid (bool), checks (list of individual results),
        and errors (list of strings describing failures).
    """
    checks = []
    errors = []

    # 1. Origin exists
    orig_result = validate_station_exists(origin)
    checks.append({"check": "origin_exists", **orig_result})
    if not orig_result["valid"]:
        errors.append(f"Origin '{origin}' not found in Metro network")

    # 2. Destination exists
    dest_result = validate_station_exists(destination)
    checks.append({"check": "destination_exists", **dest_result})
    if not dest_result["valid"]:
        errors.append(f"Destination '{destination}' not found in Metro network")

    # 3. Direct line check
    if claimed_line and orig_result["valid"] and dest_result["valid"]:
        orig_on_line = validate_station_on_line(origin, claimed_line)
        dest_on_line = validate_station_on_line(destination, claimed_line)
        direct_ok = orig_on_line and dest_on_line
        checks.append({
            "check": "direct_line",
            "line": claimed_line,
            "origin_on_line": orig_on_line,
            "destination_on_line": dest_on_line,
            "valid": direct_ok,
        })
        if not direct_ok:
            errors.append(
                f"Direct route on '{claimed_line}' invalid: "
                f"origin_on_line={orig_on_line}, dest_on_line={dest_on_line}"
            )

    # 4. Transfer check
    if claimed_transfer and claimed_transfer_lines:
        tf_result = validate_transfer_point(
            claimed_transfer, claimed_transfer_lines[0], claimed_transfer_lines[1]
        )
        checks.append({"check": "transfer_point", **tf_result})
        if not tf_result["valid"]:
            errors.append(
                f"Transfer at '{claimed_transfer}' between "
                f"{claimed_transfer_lines} is invalid"
            )

    route_valid = len(errors) == 0 and len(checks) > 0
    return {"route_valid": route_valid, "checks": checks, "errors": errors}


def validate_response_route_facts(
    response_text: str, expected_facts: list[str]
) -> dict[str, Any]:
    """Scan a response for transport route facts and validate them.

    Uses regex to find station names, line references, and transfer
    mentions in the response text, then validates each against the
    canonical metro data.

    Args:
        response_text: The LLM-generated response to validate.
        expected_facts: List of expected fact strings from the dataset.

    Returns:
        Dict with stations_mentioned (list), lines_mentioned (list),
        station_validity (dict), and facts_score (float 0-1).
    """
    text_lower = response_text.lower()

    # Find metro lines mentioned
    line_names = ["amarela", "azul", "verde", "vermelha",
                  "yellow", "blue", "green", "red"]
    line_map = {"yellow": "amarela", "blue": "azul",
                "green": "verde", "red": "vermelha"}
    lines_found = []
    for ln in line_names:
        if ln in text_lower:
            canonical = line_map.get(ln, ln)
            if canonical not in lines_found:
                lines_found.append(canonical)

    # Find station names mentioned
    stations_found = []
    station_validity = {}
    for station_key in METRO_STATIONS:
        if _normalize(station_key) in _normalize(response_text):
            if station_key not in stations_found:
                stations_found.append(station_key)
                result = validate_station_exists(station_key)
                station_validity[station_key] = result["valid"]

    # Score: fraction of expected_facts keywords found in response
    facts_matched = 0
    for fact in expected_facts:
        # Check if key terms from the fact appear in the response
        keywords = [w for w in fact.lower().split() if len(w) > 3]
        if keywords:
            matched = sum(1 for kw in keywords if kw in text_lower)
            if matched / len(keywords) >= 0.5:
                facts_matched += 1

    facts_score = facts_matched / len(expected_facts) if expected_facts else 1.0

    return {
        "stations_mentioned": stations_found,
        "lines_mentioned": lines_found,
        "station_validity": station_validity,
        "facts_matched": facts_matched,
        "facts_total": len(expected_facts),
        "facts_score": round(facts_score, 3),
    }


# -----------------------------------------------------------------------
# Self-Tests
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("\033[1m=== Transport Route Validator Self-Tests ===\033[0m\n")

    # Test 1: Station existence
    r = validate_station_exists("Marquês de Pombal")
    assert r["valid"], "Marquês de Pombal should exist"
    assert "amarela" in r["lines"] and "azul" in r["lines"]
    print("\033[1;32m[PASS]\033[0m Station existence: Marquês de Pombal")

    # Test 2: Non-existent station
    r = validate_station_exists("Hogwarts")
    assert not r["valid"], "Hogwarts should not exist"
    print("\033[1;32m[PASS]\033[0m Station non-existence: Hogwarts")

    # Test 3: Station on correct line
    assert validate_station_on_line("Aeroporto", "vermelha")
    print("\033[1;32m[PASS]\033[0m Station on line: Aeroporto on vermelha")

    # Test 4: Station NOT on wrong line
    assert not validate_station_on_line("Aeroporto", "azul")
    print("\033[1;32m[PASS]\033[0m Station not on line: Aeroporto not on azul")

    # Test 5: Valid transfer
    r = validate_transfer_point("Campo Grande", "amarela", "verde")
    assert r["valid"], "Campo Grande should connect amarela and verde"
    print("\033[1;32m[PASS]\033[0m Transfer point: Campo Grande (amarela <-> verde)")

    # Test 6: Invalid transfer
    r = validate_transfer_point("Rato", "amarela", "azul")
    assert not r["valid"], "Rato only serves amarela"
    print("\033[1;32m[PASS]\033[0m Invalid transfer: Rato cannot connect amarela-azul")

    # Test 7: Full route validation (T14 from dataset)
    r = validate_metro_route(
        origin="Odivelas",
        destination="Telheiras",
        claimed_transfer="Campo Grande",
        claimed_transfer_lines=("amarela", "verde"),
    )
    assert r["route_valid"], f"Odivelas->Telheiras via Campo Grande should be valid: {r}"
    print("\033[1;32m[PASS]\033[0m Full route: Odivelas -> Telheiras via Campo Grande")

    # Test 8: Full route validation (T02 from dataset)
    r = validate_metro_route(
        origin="Baixa-Chiado",
        destination="Aeroporto",
        claimed_transfer="Alameda",
        claimed_transfer_lines=("verde", "vermelha"),
    )
    assert r["route_valid"], f"Baixa-Chiado->Aeroporto via Alameda should be valid: {r}"
    print("\033[1;32m[PASS]\033[0m Full route: Baixa-Chiado -> Aeroporto via Alameda")

    # Test 9: Invalid route (wrong transfer)
    r = validate_metro_route(
        origin="Rato",
        destination="Aeroporto",
        claimed_transfer="Rossio",
        claimed_transfer_lines=("amarela", "vermelha"),
    )
    assert not r["route_valid"], "Rossio cannot connect amarela-vermelha"
    print("\033[1;32m[PASS]\033[0m Invalid route: bad transfer at Rossio")

    # Test 10: Direct route on single line
    r = validate_metro_route(
        origin="Cais do Sodré",
        destination="Telheiras",
        claimed_line="verde",
    )
    assert r["route_valid"], "Cais do Sodré -> Telheiras direct on verde"
    print("\033[1;32m[PASS]\033[0m Direct route: Cais do Sodré -> Telheiras on verde")

    print("\n\033[1;32m=== All 10 tests passed ===\033[0m")
