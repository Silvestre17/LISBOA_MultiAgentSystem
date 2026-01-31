# ==========================================================================
# Master Thesis - Utilities Package
#   - André Filipe Gomes Silvestre, 20240502
# ==========================================================================

from agent.utils.optimization import (
    # HTTP Session Pooling
    HTTPSessionPool,
    http_pool,
    get_cached_session,
    
    # Caching
    TTLCache,
    weather_cache,
    transport_cache,
    static_cache,
    cached,
    
    # Parallel Execution
    execute_tools_parallel,
    
    # Latency Tracking
    LatencyTracker,
    latency_tracker,
    track_latency,
    
    # Optimized Fetching
    fetch_json_optimized,
)

__all__ = [
    "HTTPSessionPool",
    "http_pool",
    "get_cached_session",
    "TTLCache",
    "weather_cache",
    "transport_cache",
    "static_cache",
    "cached",
    "execute_tools_parallel",
    "LatencyTracker",
    "latency_tracker",
    "track_latency",
    "fetch_json_optimized",
]
