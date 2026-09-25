# ==========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
#
# Shared utility functions for the tools package.
# Consolidates duplicate helpers (haversine distance, HTTP retry) used
# across multiple transport and data API modules.
# ==========================================================================

import logging
import math
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)

LISBON_TZ = ZoneInfo("Europe/Lisbon")


def lisbon_now() -> datetime:
    """Return the current Lisbon local time as a naive datetime.

    Schedule data (GTFS calendars, IPMA forecast days, event date windows) is
    authoritative for Europe/Lisbon local dates. ``datetime.now()`` uses the
    host clock, which is wrong on UTC-configured deployments around midnight
    and across DST changes. The tzinfo is stripped so existing naive-datetime
    comparisons keep working.

    Returns:
        Naive datetime carrying the current Europe/Lisbon wall-clock time.
    """
    return datetime.now(LISBON_TZ).replace(tzinfo=None)


# "Quinta da Regaleira" is an estate, not a Thursday ("quinta-feira").
# "Na quinta da próxima semana" is still a Thursday.
_QUINTA_ESTATE_RE = re.compile(
    r"\bquinta\s+d(?:a|o|as|os)\s+(?!(?:pr[oó]xima|semana|seguinte|outra)\b)\S+",
    re.IGNORECASE,
)


def without_estate_names(text: str) -> str:
    """Remove "Quinta da/do ..." estate names before looking for weekday names.

    Args:
        text: User message or query.

    Returns:
        The text with each estate name replaced by a space.
    """
    return _QUINTA_ESTATE_RE.sub(" ", text or "")


LISBON_SUN_REFERENCE = (38.7223, -9.1393)


def sunrise_sunset(day: "date", latitude: float = LISBON_SUN_REFERENCE[0], longitude: float = LISBON_SUN_REFERENCE[1]) -> Optional[tuple[str, str]]:
    """Return local sunrise and sunset times (HH:MM, Lisbon time) for a date.

    Uses the NOAA solar-position approximation (accurate to about a minute
    at Lisbon's latitude), so no external service is needed.

    Args:
        day: Calendar date.
        latitude: Latitude in degrees (default: central Lisbon).
        longitude: Longitude in degrees, east positive (default: central Lisbon).

    Returns:
        ``(sunrise, sunset)`` as ``HH:MM`` strings, or ``None`` when the sun
        does not rise or set on that date at that latitude.
    """
    gamma = 2 * math.pi / 365 * (day.timetuple().tm_yday - 1)
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    latitude_rad = math.radians(latitude)
    cos_hour_angle = (
        math.cos(math.radians(90.833)) / (math.cos(latitude_rad) * math.cos(declination))
        - math.tan(latitude_rad) * math.tan(declination)
    )
    if not -1.0 <= cos_hour_angle <= 1.0:
        return None
    hour_angle = math.degrees(math.acos(cos_hour_angle))
    midnight_utc = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)

    def _local(minutes_utc: float) -> str:
        return (midnight_utc + timedelta(minutes=minutes_utc)).astimezone(LISBON_TZ).strftime("%H:%M")

    return (
        _local(720 - 4 * (longitude + hour_angle) - equation_of_time),
        _local(720 - 4 * (longitude - hour_angle) - equation_of_time),
    )


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two GPS points on Earth.

    Args:
        lat1: Latitude of the first point in decimal degrees.
        lon1: Longitude of the first point in decimal degrees.
        lat2: Latitude of the second point in decimal degrees.
        lon2: Longitude of the second point in decimal degrees.

    Returns:
        Distance in kilometres.
    """
    R = 6371.0  # Earth's mean radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_json_with_retry(
    url: str,
    timeout: int = 15,
    max_retries: int = 3,
    backoff: float = 2.0,
    headers: Optional[dict] = None,
) -> Optional[Any]:
    """Fetch JSON from a URL with exponential-backoff retry logic.

    Args:
        url: The URL to request.
        timeout: Per-request timeout in seconds (default 15).
        max_retries: Total number of attempts (default 3).
        backoff: Exponential backoff base in seconds (default 2.0).
        headers: Optional HTTP headers dict.

    Returns:
        Parsed JSON payload, or None if all attempts fail.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            wait = backoff ** attempt
            logger.warning("Timeout on %s (attempt %d/%d). Retrying in %.1fs...",
                           url, attempt + 1, max_retries, wait)
            if attempt < max_retries - 1:
                time.sleep(wait)
        except requests.exceptions.RequestException as exc:
            wait = backoff ** attempt
            logger.warning("Request error on %s: %s. Retrying in %.1fs...",
                           url, exc, wait)
            if attempt < max_retries - 1:
                time.sleep(wait)
        except ValueError:
            logger.error("Invalid JSON response from %s", url)
            return None
    return None


# ==========================================================================
# Test Block
# ==========================================================================


if __name__ == "__main__":
    """Run utility smoke checks when this module is executed directly."""

    import json

    def _run_test(test_name: str, fn, *args, **kwargs) -> bool:
        """Run a single callable and print pass/fail output."""
        try:
            result = fn(*args, **kwargs)
            print(f"PASS: {test_name}")
            print(json.dumps(result, ensure_ascii=False))
            return True
        except Exception as exc:
            print(f"FAIL: {test_name} -> {exc}")
            return False

    print("=== utils.py smoke tests ===")
    passed = 0
    total = 0

    total += 1
    if _run_test(
        "haversine_distance(Lisbon center to Belém)",
        haversine_distance,
        38.7169,
        -9.1396,
        38.6965,
        -9.2045,
    ):
        passed += 1

    total += 1
    if _run_test(
        "fetch_json_with_retry(empty endpoint) returns None",
        fetch_json_with_retry,
        "https://example.com/does-not-exist-xyz",
        timeout=1,
        max_retries=1,
    ):
        passed += 1

    print(f"utils.py smoke tests completed: {passed}/{total}")
