# ==========================================================================
# Master Thesis - Resilient IPMA HTTP Client
#   - André Filipe Gomes Silvestre, 20240502
#
# Process-local cache and recovery for official IPMA weather resources.
# ==========================================================================
"""Fetch validated IPMA data without treating unavailable warnings as evidence."""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError, ReadTimeoutError

logger = logging.getLogger(__name__)
IPMA_HOST = "api.ipma.pt"
WARNINGS_URL = f"https://{IPMA_HOST}/open-data/forecast/warnings/warnings_www.json"
PUBLIC_WARNINGS_URL = f"https://{IPMA_HOST}/public-data/warnings/warnings_www.json"
DAILY_PATH = re.compile(r"/open-data/forecast/meteorology/cities/daily/(\d+)\.json$")
LISBON_TZ = ZoneInfo("Europe/Lisbon")
TRANSIENT_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class FetchResult:
    """A payload and its provenance; missing data never means no warnings."""

    data: Any = None
    retrieved_at: datetime | None = None
    source_updated_at: str | None = None
    stale: bool = False
    source: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class _CacheEntry:
    result: FetchResult
    stored_at: float


def _parse_timestamp(value: Any) -> datetime:
    """Parse a provider timestamp, interpreting unqualified IPMA times as UTC."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing source timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if "T" not in value and " " not in value:
        raise ValueError("source timestamp must include time")
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _validate_source_age(updated: str | None, now: datetime, max_age: float) -> None:
    """Reject timestamps beyond the supported freshness policy."""
    if updated is None:
        return
    age = (now - _parse_timestamp(updated)).total_seconds()
    if age > max_age or age < -300:
        raise ValueError("source timestamp outside freshness policy")


def _number(value: Any, field: str) -> float:
    """Reject non-numeric and non-finite forecast values."""
    if isinstance(value, bool):
        raise ValueError(f"invalid {field}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"invalid {field}")
    return number


def _validate_rows(rows: Any, today: date, *, warnings: bool) -> list[dict[str, Any]]:
    """Validate complete records before the caller interprets their contents."""
    if not isinstance(rows, list) or (not warnings and not rows):
        raise ValueError("invalid data list")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid record")
        if warnings:
            if row.get("awarenessLevelID") not in {"green", "yellow", "orange", "red"}:
                raise ValueError("invalid warning level")
            if not row.get("idAreaAviso") or not row.get("awarenessTypeName"):
                raise ValueError("missing warning area or type")
            start = _parse_timestamp(row.get("startTime"))
            end = _parse_timestamp(row.get("endTime"))
            if end < start:
                raise ValueError("inverted warning interval")
        else:
            date.fromisoformat(row.get("forecastDate", ""))
            minimum = _number(row.get("tMin"), "tMin")
            maximum = _number(row.get("tMax"), "tMax")
            probability = _number(row.get("precipitaProb"), "precipitaProb")
            # -99 is the provider's explicit unavailable-value sentinel.
            if minimum != -99 and maximum != -99 and minimum > maximum:
                raise ValueError("inverted temperature range")
            if probability != -99 and not 0 <= probability <= 100:
                raise ValueError("invalid precipitation probability")
            _number(row.get("idWeatherType"), "idWeatherType")
            _number(row.get("classWindSpeed"), "classWindSpeed")
            if not isinstance(row.get("predWindDir"), str):
                raise ValueError("missing wind direction")
    if not warnings and not any(date.fromisoformat(row["forecastDate"]) >= today for row in rows):
        raise ValueError("forecast contains no current or future day")
    return rows


def _normalize(url: str, payload: Any, now: datetime) -> tuple[Any, str | None]:
    """Normalize official alternate schemas into the existing tool schema."""
    today = now.astimezone(LISBON_TZ).date()
    if url in {WARNINGS_URL, PUBLIC_WARNINGS_URL}:
        updated = None
        if url == PUBLIC_WARNINGS_URL:
            if not isinstance(payload, dict):
                raise ValueError("invalid public warnings object")
            updated = payload.get("dataAtualiza")
            _parse_timestamp(updated)
            payload = payload.get("data")
        rows = _validate_rows(payload, today, warnings=True)
        current = [row for row in rows if _parse_timestamp(row["endTime"]) > now]
        if rows and not current:
            raise ValueError("warning feed contains only expired intervals")
        return current, updated
    if "/public-data/forecast/aggregate/" in url:
        if not isinstance(payload, list):
            raise ValueError("invalid aggregate forecast list")
        location = int(url.rsplit("/", 1)[1].split(".")[0])
        rows = []
        timestamps = []
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError("invalid aggregate forecast record")
            if str(row.get("idPeriodo")) != "24":
                continue
            if int(row.get("globalIdLocal", -1)) != location:
                raise ValueError("unexpected forecast location")
            timestamp = row.get("dataUpdate")
            _parse_timestamp(timestamp)
            timestamps.append(timestamp)
            rows.append({
                "forecastDate": row.get("dataPrev", "").split("T")[0].split(" ")[0],
                "tMin": row.get("tMin"), "tMax": row.get("tMax"),
                "precipitaProb": row.get("probabilidadePrecipita"),
                "predWindDir": row.get("ddVento"),
                "idWeatherType": row.get("idTipoTempo"),
                "classWindSpeed": row.get("idFfxVento"),
                "classPrecInt": row.get("idIntPrecipita", -99),
            })
        _validate_rows(rows, today, warnings=False)
        # The oldest contributing update conservatively describes the snapshot.
        updated = min(timestamps, key=_parse_timestamp)
        return {"globalIdLocal": location, "dataUpdate": updated, "data": rows}, updated
    if not isinstance(payload, dict) or payload.get("error"):
        raise ValueError("invalid forecast object")
    updated = payload.get("dataUpdate")
    _parse_timestamp(updated)
    rows = payload.get("data")
    if "hp-daily-forecast-day" in url and isinstance(rows, list):
        rows = [{"forecastDate": payload.get("forecastDate"), **row}
                if isinstance(row, dict) else row for row in rows]
    rows = _validate_rows(rows, today, warnings=False)
    match = DAILY_PATH.search(urlsplit(url).path)
    if match and str(payload.get("globalIdLocal")) != match.group(1):
        raise ValueError("unexpected forecast location")
    return {**payload, "data": rows}, updated


class IPMAClient:
    """Share successful snapshots and suppress repeated requests during outages.

    Args:
        session: Optional injected HTTP session for deterministic tests.
        connect_timeout: Maximum connection wait per attempt, in seconds.
        read_timeout: Maximum idle read wait per attempt, in seconds.
        budget: Best-effort budget for a request sequence, in seconds.
        cache_ttl: Successful snapshot reuse interval, in seconds.
        forecast_max_age: Maximum retrieval age for stale forecasts, in seconds.
        cooldown: Pause after consecutive connection failures, in seconds.
        failure_threshold: Connection failures needed to activate the host pause.
        warnings_source_max_age: Maximum age of a warning source timestamp.
        forecast_source_max_age: Maximum age of a forecast source timestamp.
        clock: Monotonic clock, injectable for tests.
        now: UTC wall clock, injectable for tests.

    Requests does not guarantee a hard deadline for DNS or streamed response
    bodies. The budget bounds scheduled attempts and their timeout allocation.
    Sessions are per-thread because requests.Session is not a concurrency API.
    """

    def __init__(
        self, *, session: requests.Session | None = None,
        connect_timeout: float = 3, read_timeout: float = 5,
        budget: float = 10, cache_ttl: float = 300,
        forecast_max_age: float = 3600, cooldown: float = 30,
        failure_threshold: int = 2, clock: Callable[[], float] = time.monotonic,
        warnings_source_max_age: float = 3600, forecast_source_max_age: float = 86400,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if min(connect_timeout, read_timeout, budget, cache_ttl, forecast_max_age,
               cooldown, warnings_source_max_age, forecast_source_max_age) <= 0:
            raise ValueError("timeouts and cache intervals must be positive")
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        self._injected_session = session
        self._threads = threading.local()
        self._lock = threading.Lock()
        self._resource_locks: dict[str, threading.Lock] = {}
        self._cache: dict[str, _CacheEntry] = {}
        self._latest: dict[str, tuple[int, FetchResult]] = {}
        self._failures = 0
        self._paused_until = 0.0
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._budget = budget
        self._cache_ttl = cache_ttl
        self._forecast_max_age = forecast_max_age
        self._cooldown = cooldown
        self._failure_threshold = failure_threshold
        self._warnings_source_max_age = warnings_source_max_age
        self._forecast_source_max_age = forecast_source_max_age
        self._clock = clock
        self._now = now

    def _session(self) -> requests.Session:
        """Return a dedicated session with implicit retries disabled."""
        if self._injected_session is not None:
            return self._injected_session
        if not hasattr(self._threads, "session"):
            session = requests.Session()
            session.mount("https://", HTTPAdapter(max_retries=0))
            session.headers["User-Agent"] = "LISBOA/IPMA weather client"
            self._threads.session = session
        return self._threads.session

    def _cached(self, url: str, *, stale: bool = False, error: str | None = None) -> FetchResult | None:
        """Return a fresh snapshot, or a bounded stale forecast on failure."""
        entry = self._cache.get(url)
        if entry is None:
            return None
        age = self._clock() - entry.stored_at
        limit = self._forecast_max_age if stale else self._cache_ttl
        is_warnings = url in {WARNINGS_URL, PUBLIC_WARNINGS_URL}
        if age >= limit or (stale and is_warnings):
            return None
        try:
            _validate_source_age(entry.result.source_updated_at, self._now(),
                                 self._warnings_source_max_age if is_warnings
                                 else self._forecast_source_max_age)
        except ValueError:
            return None
        data = entry.result.data
        if is_warnings:
            data = [row for row in data if _parse_timestamp(row["endTime"]) > self._now()]
            if entry.result.data and not data:
                return None
        if isinstance(entry.result.data, dict):
            today = self._now().astimezone(LISBON_TZ).date()
            if not any(date.fromisoformat(row["forecastDate"]) >= today
                       for row in entry.result.data["data"]):
                return None
        return replace(entry.result, data=data, stale=stale, error=error)

    def fetch(self, url: str, use_cache: bool = True) -> FetchResult:
        """Fetch one official resource with single-flight cache and recovery.

        Args:
            url: Official HTTPS open-data URL.
            use_cache: Reuse fresh data; overlapping refreshes still share work.

        Returns:
            Validated data and provenance, or an explicit unavailable result.
        """
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != IPMA_HOST:
            return FetchResult(error="unsupported IPMA URL")
        with self._lock:
            resource_lock = self._resource_locks.setdefault(url, threading.Lock())
            generation = self._latest.get(url, (0, None))[0]
        with resource_lock:
            with self._lock:
                latest = self._latest.get(url)
            if latest is not None and latest[0] > generation:
                return latest[1]
            cached = self._cached(url)
            if cached is not None and use_cache:
                return cached
            with self._lock:
                paused = self._clock() < self._paused_until
            if paused:
                error = "IPMA host temporarily paused after repeated failures"
                return self._cached(url, stale=True, error=error) or FetchResult(error=error)
            result, connection_failed = self._request(url)
            with self._lock:
                if not connection_failed:
                    self._failures = 0
                    self._paused_until = 0.0
                else:
                    self._failures += 1
                    if self._failures >= self._failure_threshold:
                        self._paused_until = self._clock() + self._cooldown
            if result.data is not None:
                self._cache[url] = _CacheEntry(result, self._clock())
            else:
                result = self._cached(url, stale=True, error=result.error) or result
            with self._lock:
                self._latest[url] = (generation + 1, result)
            return result

    def _request(self, url: str) -> tuple[FetchResult, bool]:
        """Try a primary resource and compatible fallback within one budget."""
        urls = [url]
        match = DAILY_PATH.search(urlsplit(url).path)
        if url == WARNINGS_URL:
            urls.append(PUBLIC_WARNINGS_URL)
        elif match:
            urls.append(f"https://{IPMA_HOST}/public-data/forecast/aggregate/{match.group(1)}.json")
        deadline = self._clock() + self._budget
        error = "IPMA request budget exhausted"
        for target in urls:
            for attempt in range(2):
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return FetchResult(error=error), False
                # Sum of the allocated connect/read waits stays within budget.
                connect = min(self._connect_timeout, remaining / 2)
                read = min(self._read_timeout, remaining - connect)
                started = self._clock()
                retry = False
                response = None
                try:
                    response = self._session().get(target, timeout=(connect, read), allow_redirects=False)
                    if 300 <= response.status_code < 400:
                        raise ValueError("unexpected IPMA redirect")
                    response.raise_for_status()
                    data, updated = _normalize(target, response.json(), self._now())
                    if updated is None and response.headers.get("Last-Modified"):
                        updated = parsedate_to_datetime(response.headers["Last-Modified"]).isoformat()
                    _validate_source_age(updated, self._now(),
                                         self._warnings_source_max_age if url in {WARNINGS_URL, PUBLIC_WARNINGS_URL}
                                         else self._forecast_source_max_age)
                    return FetchResult(data, self._now(), updated, source=target), False
                except requests.RequestException as exc:
                    first_cause = exc.args[0] if exc.args else None
                    reason = getattr(first_cause, "reason", first_cause)
                    connection_failed = isinstance(exc, (requests.ConnectTimeout, requests.exceptions.SSLError)) or (
                        isinstance(reason, (ConnectTimeoutError, NewConnectionError)))
                    if connection_failed:
                        # A path change cannot recover failed TCP/TLS setup.
                        error = f"{type(exc).__name__}: IPMA connection unavailable"
                        logger.warning("IPMA %s attempt=%s elapsed=%.3fs error=%s", target, attempt + 1,
                                       self._clock() - started, type(exc).__name__)
                        return FetchResult(error=error), True
                    error = f"{type(exc).__name__}: IPMA request unavailable"
                    retry = isinstance(exc, requests.ReadTimeout) or isinstance(reason, ReadTimeoutError) or (
                        response is not None and response.status_code in TRANSIENT_STATUS)
                except (ValueError, TypeError, KeyError, OverflowError) as exc:
                    error = f"Invalid IPMA payload: {exc}"
                finally:
                    if response is not None:
                        response.close()
                logger.warning("IPMA %s attempt=%s elapsed=%.3fs error=%s", target, attempt + 1,
                               self._clock() - started, error)
                if not retry:
                    break
        return FetchResult(error=error), False


IPMA_CLIENT = IPMAClient()
