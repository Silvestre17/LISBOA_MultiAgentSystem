# ==========================================================================
# Master Thesis - Optimization Utilities
#   - André Filipe Gomes Silvestre, 20240502
#
#   Performance optimization utilities for the Lisbon Urban Assistant.
#   Features:
#     - HTTP Session Pooling (connection reuse)
#     - API Response Caching with TTL
#     - Parallel Tool Execution
#     - Request Timeouts
# ==========================================================================

import hashlib
import json
import time
from concurrent.futures import as_completed
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from agent.utils.langsmith_tracing import ContextThreadPoolExecutor

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
            # Set default timeout
            self._session.timeout = (3, 10)  # (connect, read) timeouts
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


def get_cached_session() -> requests.Session:
    """Returns the shared HTTP session for connection reuse."""
    return http_pool.session


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

    def _make_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Creates a cache key from function signature."""
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

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
static_cache = TTLCache(default_ttl=3600)     # 1 hour for static data


def cached(cache: TTLCache = None, ttl: int = None):
    """
    Decorator for caching function results.

    Usage:
        @cached(cache=weather_cache, ttl=300)
        def get_weather_data(location):
            ...
    """
    if cache is None:
        cache = TTLCache(default_ttl=ttl or 60)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = cache._make_key(func.__name__, args, kwargs)

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result

        # Add method to bypass cache
        wrapper.uncached = func
        return wrapper

    return decorator


# ==========================================================================
# Parallel Tool Execution
# ==========================================================================

def execute_tools_parallel(
    tools_to_call: List[Dict[str, Any]],
    available_tools: List,
    max_workers: int = 4,
    timeout: float = 30.0
) -> Dict[str, str]:
    """
    Execute multiple tool calls in parallel.

    Args:
        tools_to_call: List of tool call dicts with 'name' and 'args' keys.
        available_tools: List of LangChain tool objects.
        max_workers: Maximum number of parallel workers.
        timeout: Maximum time to wait for all tools.

    Returns:
        Dict mapping tool call IDs to their results.
    """
    if not tools_to_call:
        return {}

    # Create tool name to object mapping
    tool_map = {tool.name: tool for tool in available_tools}

    results = {}

    def execute_single_tool(tool_call: Dict) -> Tuple[str, str]:
        tool_name = tool_call.get('name', '')
        tool_args = tool_call.get('args', {})
        tool_id = tool_call.get('id', f'call_{hash(tool_name)}')

        if tool_name not in tool_map:
            return (tool_id, f"Tool '{tool_name}' not found.")

        try:
            result = tool_map[tool_name].invoke(tool_args)
            return (tool_id, str(result))
        except Exception as e:
            return (tool_id, f"Error executing {tool_name}: {str(e)}")

    # Limit workers to number of tools
    num_workers = min(max_workers, len(tools_to_call))

    with ContextThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_tool = {
            executor.submit(execute_single_tool, tc): tc
            for tc in tools_to_call
        }

        for future in as_completed(future_to_tool, timeout=timeout):
            try:
                tool_id, result = future.result()
                results[tool_id] = result
            except Exception as e:
                tool_call = future_to_tool[future]
                tool_id = tool_call.get('id', 'unknown')
                results[tool_id] = f"Execution error: {str(e)}"

    return results


# ==========================================================================
# Latency Tracking
# ==========================================================================

class LatencyTracker:
    """
    Tracks latency metrics for performance monitoring.
    """

    def __init__(self):
        self._metrics: Dict[str, List[float]] = {}
        self._lock = Lock()

    def record(self, operation: str, latency_ms: float):
        """Record a latency measurement."""
        with self._lock:
            if operation not in self._metrics:
                self._metrics[operation] = []
            self._metrics[operation].append(latency_ms)
            # Keep only last 100 measurements
            if len(self._metrics[operation]) > 100:
                self._metrics[operation] = self._metrics[operation][-100:]

    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        with self._lock:
            if operation not in self._metrics or not self._metrics[operation]:
                return {'count': 0, 'avg': 0, 'min': 0, 'max': 0, 'p95': 0}

            latencies = sorted(self._metrics[operation])
            count = len(latencies)
            avg = sum(latencies) / count
            p95_idx = int(count * 0.95)

            return {
                'count': count,
                'avg': round(avg, 2),
                'min': round(min(latencies), 2),
                'max': round(max(latencies), 2),
                'p95': round(latencies[p95_idx] if p95_idx < count else latencies[-1], 2)
            }


# Global latency tracker
latency_tracker = LatencyTracker()


def track_latency(operation: str):
    """
    Decorator to track function execution latency.

    Usage:
        @track_latency("weather_api_call")
        def fetch_weather():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                latency_ms = (time.time() - start) * 1000
                latency_tracker.record(operation, latency_ms)
        return wrapper
    return decorator


# ==========================================================================
# Request Helpers with Optimization
# ==========================================================================

def fetch_json_optimized(
    url: str,
    cache: TTLCache = None,
    cache_ttl: int = 60,
    timeout: float = 10.0
) -> Optional[Dict[str, Any]]:
    """
    Fetch JSON with connection pooling and optional caching.

    Args:
        url: URL to fetch.
        cache: Optional cache instance.
        cache_ttl: Cache TTL in seconds.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON or None on error.
    """
    # Check cache first
    if cache is not None:
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result

    try:
        response = http_pool.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        # Cache the result
        if cache is not None:
            cache.set(cache_key, data, cache_ttl)

        return data
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.RequestException:
        return None
    except ValueError:
        return None


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

    # Test fetch_json_optimized
    print("\n🔄 Testing Optimized JSON Fetch...")
    url = "https://api.ipma.pt/open-data/distrits-islands.json"

    # First call (not cached)
    start = time.time()
    data1 = fetch_json_optimized(url, cache=static_cache, cache_ttl=60)
    print(f"   First call: {(time.time() - start) * 1000:.0f}ms (network)")

    # Second call (cached)
    start = time.time()
    data2 = fetch_json_optimized(url, cache=static_cache, cache_ttl=60)
    print(f"   Second call: {(time.time() - start) * 1000:.0f}ms (cached)")

    print("\n✅ All optimization utilities working!")
