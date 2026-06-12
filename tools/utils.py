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
import time
from datetime import datetime
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
