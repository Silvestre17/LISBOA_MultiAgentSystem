# ==========================================================================
# Master Thesis - IPMA HTTP Client
#   - André Filipe Gomes Silvestre, 20240502
#
# Short-lived cache and optional relay for IPMA open-data resources.
# ==========================================================================
"""Fetch IPMA open data without treating an unavailable feed as evidence.

The IPMA network does not accept connections from some cloud providers,
including the AWS addresses used by Hugging Face Spaces. ``IPMA_BASE_URL``
points those deployments at a read-only relay that serves the same paths;
locally it stays unset and requests go to the official host.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable

import requests

logger = logging.getLogger(__name__)

IPMA_ORIGIN = "https://api.ipma.pt"
WARNINGS_URL = f"{IPMA_ORIGIN}/open-data/forecast/warnings/warnings_www.json"


@dataclass(frozen=True)
class FetchResult:
    """A payload and its provenance; missing data never means no warnings."""

    data: Any = None
    retrieved_at: datetime | None = None
    stale: bool = False
    error: str | None = None


def _is_valid(url: str, payload: Any) -> bool:
    """Accept only the shapes the tools read: a warning list or a forecast object."""
    if url == WARNINGS_URL:
        return isinstance(payload, list)
    return isinstance(payload, dict) and isinstance(payload.get("data"), list) and bool(payload["data"])


class IPMAClient:
    """Cache successful IPMA responses and keep outages short for the user.

    Args:
        base_url: Host serving IPMA paths; defaults to ``IPMA_BASE_URL`` or IPMA.
        timeout: Connect and read timeouts per request, in seconds.
        cache_ttl: Reuse interval for a successful response, in seconds.
        stale_ttl: Maximum age of a forecast served after a failed refresh.
            Warnings are never served stale.
        cooldown: Pause after a connection failure, so one question does not
            wait for the same unreachable host once per tool call.
        session: Optional HTTP session, injectable for tests.
        clock: Monotonic clock, injectable for tests.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: tuple[float, float] = (3, 6),
        cache_ttl: float = 300,
        stale_ttl: float = 6 * 3600,
        cooldown: float = 15,
        session: requests.Session | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = (base_url or os.getenv("IPMA_BASE_URL") or IPMA_ORIGIN).rstrip("/")
        self._timeout = timeout
        self._cache_ttl = cache_ttl
        self._stale_ttl = stale_ttl
        self._cooldown = cooldown
        self._clock = clock
        self._session = session or requests.Session()
        self._session.headers["User-Agent"] = "LISBOA-Thesis-Agent/1.0 (IPMA open data)"
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, FetchResult]] = {}
        self._unreachable_until = 0.0

    @property
    def uses_relay(self) -> bool:
        """Return whether requests go through a relay instead of IPMA itself."""
        return self.base_url != IPMA_ORIGIN

    def fetch(self, url: str) -> FetchResult:
        """Return fresh data, a bounded stale forecast, or an explicit failure.

        Args:
            url: Official IPMA open-data URL, as documented at api.ipma.pt.

        Returns:
            The validated payload with its retrieval time, or an error.
        """
        if not url.startswith(f"{IPMA_ORIGIN}/open-data/"):
            return FetchResult(error="unsupported IPMA URL")
        with self._lock:
            stored_at, cached = self._cache.get(url, (None, None))
            unreachable = self._clock() < self._unreachable_until
        age = None if stored_at is None else self._clock() - stored_at
        if cached is not None and age < self._cache_ttl:
            return cached

        result = FetchResult(error="IPMA unreachable; retrying shortly") if unreachable else self._download(url)
        if result.data is not None:
            with self._lock:
                self._cache[url] = (self._clock(), result)
            return result
        if cached is not None and url != WARNINGS_URL and age < self._stale_ttl:
            return replace(cached, stale=True)
        return result

    def _download(self, url: str) -> FetchResult:
        """Request one resource; a host that stays unreachable starts the cooldown."""
        target = self.base_url + url[len(IPMA_ORIGIN):]
        started = self._clock()
        for attempt in range(2):
            try:
                response = self._session.get(target, timeout=self._timeout)
                response.raise_for_status()
                payload = response.json()
            except requests.ConnectionError as exc:
                if attempt == 0 and not isinstance(exc, requests.ConnectTimeout):
                    # A pooled keep-alive connection that the relay or a NAT
                    # already closed fails at once; retry on a new connection.
                    self._session.close()
                    continue
                with self._lock:
                    self._unreachable_until = self._clock() + self._cooldown
                error, detail = type(exc).__name__, exc
            except (requests.RequestException, ValueError) as exc:
                error, detail = type(exc).__name__, exc
            else:
                if _is_valid(url, payload):
                    return FetchResult(payload, datetime.now(timezone.utc))
                error, detail = "unexpected payload", type(payload).__name__
            break
        logger.warning("IPMA %s failed after %.2fs: %s (%s)", target, self._clock() - started, error, detail)
        return FetchResult(error=error)


IPMA_CLIENT = IPMAClient()
