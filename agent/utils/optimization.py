# ==========================================================================
# Master Thesis - Optimization Utilities
#   - André Filipe Gomes Silvestre, 20240502
#
#   Performance optimization utilities for the Lisbon Urban Assistant.
#   Features:
#     - HTTP Session Pooling (connection reuse)
#     - API Response Caching with TTL
# ==========================================================================

import time
from threading import Lock
from typing import Any, Dict, Optional, Tuple

import requests

# ==========================================================================
# HTTP Session Pooling
# ==========================================================================


class HTTPSessionPool:
    """
    Thread-safe HTTP session pool for connection reuse.

    Reusing HTTP connections significantly reduces latency by avoiding
    TCP/TLS handshake overhead on every request.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._session = None
        return cls._instance

    @property
    def session(self) -> requests.Session:
        """Returns a shared requests Session with optimized settings."""
        if self._session is None:
            self._session = requests.Session()
            # Configure connection pooling
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,  # Number of connection pools
                pool_maxsize=20,       # Max connections per pool
                max_retries=2,
                pool_block=False
            )
            self._session.mount('http://', adapter)
            self._session.mount('https://', adapter)
        return self._session

    def get(self, url: str, **kwargs) -> requests.Response:
        """GET request with connection reuse."""
        if 'timeout' not in kwargs:
            kwargs['timeout'] = (3, 10)
        return self.session.get(url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        """POST request with connection reuse."""
        if 'timeout' not in kwargs:
            kwargs['timeout'] = (3, 10)
        return self.session.post(url, **kwargs)


# Global session pool instance
http_pool = HTTPSessionPool()


# ==========================================================================
# API Response Caching
# ==========================================================================

class TTLCache:
    """
    Thread-safe cache with Time-To-Live (TTL) support.

    Caches API responses to avoid redundant network calls for
    data that doesn't change frequently (weather, transport status, etc.)
    """

    def __init__(self, default_ttl: int = 60):
        """
        Initialize cache with default TTL.

        Args:
            default_ttl: Default time-to-live in seconds.
        """
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = Lock()
        self.default_ttl = default_ttl
        self._last_cleanup = 0.0
        self._cleanup_interval = min(max(default_ttl, 1), 60)

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if time.time() < expiry:
                    return value
                # Clean up expired entry
                del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cache value with TTL."""
        ttl = ttl if ttl is not None else self.default_ttl
        current_time = time.time()
        if current_time - self._last_cleanup >= self._cleanup_interval:
            self.cleanup_expired(current_time=current_time)

        expiry = current_time + ttl
        with self._lock:
            self._cache[key] = (value, expiry)

    def clear(self):
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self, current_time: Optional[float] = None):
        """Remove all expired entries."""
        current_time = current_time if current_time is not None else time.time()
        with self._lock:
            expired_keys = [
                k for k, (_, expiry) in self._cache.items()
                if current_time >= expiry
            ]
            for k in expired_keys:
                del self._cache[k]
            self._last_cleanup = current_time


# Global cache instances with different TTLs for different data types
weather_cache = TTLCache(default_ttl=300)     # 5 minutes for weather
transport_cache = TTLCache(default_ttl=60)    # 1 minute for transport


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Optimization Utilities Test")
    print("=" * 60)

    # Test HTTP Session Pool
    print("\n📡 Testing HTTP Session Pool...")
    start = time.time()
    for i in range(3):
        resp = http_pool.get("https://api.ipma.pt/open-data/distrits-islands.json")
        print(f"   Request {i + 1}: {resp.status_code} ({(time.time() - start) * 1000:.0f}ms)")

    # Test Cache
    print("\n💾 Testing TTL Cache...")
    test_cache = TTLCache(default_ttl=2)
    test_cache.set("key1", "value1")
    print(f"   Cached value: {test_cache.get('key1')}")
    time.sleep(3)
    print(f"   After TTL: {test_cache.get('key1')}")

    print("\n✅ All optimization utilities working!")
