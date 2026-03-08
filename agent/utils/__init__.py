# ==========================================================================
# Master Thesis - Utilities Package
#   - André Filipe Gomes Silvestre, 20240502
# ==========================================================================

from agent.utils.optimization import (  # HTTP Session Pooling; Caching; Parallel Execution; Latency Tracking; Optimized Fetching
    HTTPSessionPool,
    LatencyTracker,
    TTLCache,
    cached,
    execute_tools_parallel,
    fetch_json_optimized,
    get_cached_session,
    http_pool,
    latency_tracker,
    static_cache,
    track_latency,
    transport_cache,
    weather_cache,
)
from agent.utils.response_formatter import (
    ensure_response_title,
    format_response,
    generate_response_title,
)

__all__ = [
    "format_response",
    "generate_response_title",
    "ensure_response_title",
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
