# Tools API Overview

This document provides an overview of all available tools in the Lisbon Urban Assistant project.

---

## 📋 Table of Contents

1. [Weather Tools (IPMA)](#weather-tools-ipma)
2. [Transport Tools](#transport-tools)
3. [Open Data Tools (Dados Abertos)](#open-data-tools)
4. [VisitLisboa Tools](#visitlisboa-tools)
5. [Vector Store](#vector-store)
6. [Tool Categories](#tool-categories)

---

## Weather Tools (IPMA)

**Module**: `tools/ipma_api.py`  
**Data Source**: Instituto Português do Mar e da Atmosfera  
**Update Frequency**: Real-time API calls

### Available Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `get_weather_warnings()` | Active weather warnings for Lisbon | Formatted warning list |
| `get_weather_forecast()` | 1-5 day forecast | Daily forecast with temps, precipitation |
| `get_current_weather_summary()` | Combined summary | Today's weather + warnings |

**See**: [ipma_api.md](ipma_api.md) for detailed documentation.

---

## Transport Tools

**Module**: `tools/transport_api.py`  
**Data Sources**: Metro de Lisboa, Carris Metropolitana, CP  
**Update Frequency**: Real-time API calls

### Available Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `get_metro_status()` | Status of all 4 metro lines | Line-by-line operational status |
| `get_carris_alerts()` | Active bus service alerts | Alert descriptions and affected routes |
| `get_carris_stop_info()` | Bus stop details + arrivals | Stop info with real-time arrivals |
| `search_carris_lines()` | Search bus lines by name/number | Matching line information |
| `get_train_status()` | CP train delays | Train status with delay information |
| `get_transport_summary()` | All transport overview | Combined status of metro/bus/train |

**See**: [transport_api.md](transport_api.md) for detailed documentation.

---

## Open Data Tools

**Module**: `tools/dados_abertos.py`  
**Data Source**: Lisboa Aberta (dados.cm-lisboa.pt)  
**Update Frequency**: Static metadata, dynamic GeoJSON fetch

### Available Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `find_nearby_services()` | Search services by type + location | Service list with distances |
| `list_available_datasets()` | Browse all available datasets | Dataset list with descriptions |
| `get_dataset_details()` | Get dataset schema and info | Detailed dataset information |

**Key Features**:
- **Dynamic GeoJSON Fetching**: Fetches data on-demand from stable URLs
- **Proximity Filtering**: Haversine distance calculation
- **Retry Logic**: 3 retries with exponential backoff (15s timeout)

**See**: [dados_abertos.md](dados_abertos.md) for detailed documentation.

---

## VisitLisboa Tools

**Module**: `tools/visitlisboa_api.py`  
**Data Source**: Scraped from visitlisboa.com  
**Search Method**: Semantic search with ChromaDB + fallback to JSON

### Available Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `search_cultural_events()` | Search events with DATE FILTERING | Event list with dates and locations |
| `search_places_attractions()` | Semantic search for places | Place list with descriptions |
| `get_event_categories()` | List all event categories | Category counts |
| `get_place_categories()` | List all place categories | Category counts |
| `search_lisbon_knowledge()` | Search across all collections | Combined results from PDF/places/events |

**Key Features**:
- **Semantic Search**: AI-powered relevance matching
- **Date Parsing**: Natural language dates ("next week", "this weekend")
- **Date Filtering**: Critical for events (defaults to upcoming 30 days)
- **Fallback**: JSON keyword search if vector store unavailable

**See**: [visitlisboa_api.md](visitlisboa_api.md) for detailed documentation.

---

## Vector Store

**Module**: `tools/vector_store.py`  
**Technology**: ChromaDB with HuggingFace embeddings  
**Model**: BAAI/bge-m3 (multilingual)

### Available Functions

| Function | Description | Purpose |
|----------|-------------|---------|
| `sync_all()` | Incremental sync of all collections | Update vector database |
| `search()` | Semantic search across collections | Retrieve relevant documents |
| `get_stats()` | Database statistics | Monitor collection sizes |

**Collections**:
1. **lisbon_pdf**: Static PDF guide (~900 chunks)
2. **lisbon_places**: VisitLisboa places (~300 docs)
3. **lisbon_events**: VisitLisboa events (~200 docs)

**Sync Strategy**:
- **Incremental Updates**: Only processes changed documents
- **Content Hashing**: Detects modifications via SHA-256
- **Automatic Cleanup**: Removes deleted items

**See**: [vector_store.md](vector_store.md) for detailed documentation.

---

## Tool Categories

### By Update Frequency

```
Real-Time (API):
├── IPMA Weather (get_weather_*, get_current_weather_summary)
├── Metro Status (get_metro_status)
├── Carris Alerts (get_carris_alerts, get_carris_stop_info)
└── Train Status (get_train_status)

On-Demand Fetch:
└── Dados Abertos GeoJSON (find_nearby_services)

Static + Search:
├── VisitLisboa Events (search_cultural_events) [Daily scrape]
├── VisitLisboa Places (search_places_attractions) [Weekly scrape]
└── Vector Store (search_lisbon_knowledge) [Sync on changes]
```

### By Data Type

```
Temporal Data:
├── Weather Forecasts (1-5 days)
├── Events (with date ranges)
└── Transport Status (real-time)

Geospatial Data:
├── GeoJSON datasets (Dados Abertos)
├── Places (with coordinates)
└── Services (with proximity filtering)

Textual Data:
├── Event descriptions
├── Place descriptions
└── PDF guide content
```

---

## Tool Selection Guide

### For Users Asking About...

**Weather**:
- "What's the weather?" → `get_current_weather_summary()`
- "Weather next week?" → `get_weather_forecast(days=7)`
- "Any weather warnings?" → `get_weather_warnings()`

**Transport**:
- "Is metro working?" → `get_metro_status()`
- "Bus delays?" → `get_carris_alerts()`
- "Transport status?" → `get_transport_summary()`

**Places/Attractions**:
- "Best museums?" → `search_places_attractions(query="museums")`
- "Where to eat?" → `search_places_attractions(category="Restaurants")`
- "Viewpoints in Lisbon?" → `search_places_attractions(query="viewpoints")`

**Events**:
- "Events today?" → `search_cultural_events(date_filter="today")`
- "Concerts this week?" → `search_cultural_events(query="concerts", date_filter="this week")`
- "Weekend events?" → `search_cultural_events(date_filter="this weekend")`

**Services**:
- "Pharmacies nearby?" → `find_nearby_services("farmácias", lat, lon)`
- "Hospitals?" → `find_nearby_services("hospitais")`
- "WiFi hotspots?" → `find_nearby_services("wifi")`

---

## Error Handling

All tools implement:
- **Timeout Handling**: 10-15s timeouts with retries
- **Graceful Degradation**: Fallback mechanisms
- **Logging**: Structured logging for debugging
- **User-Friendly Messages**: Clear error explanations

### Common Error Responses

```python
# API Unavailable
"❌ Failed to fetch [service] data. The API may be temporarily unavailable."

# No Results
"❌ No [items] found matching: '[query]'"

# Timeout
"⚠️ Request timed out. Retrying in {wait_time}s..."
```

---

## Performance Considerations

### Response Times

| Tool Category | Avg Response Time | Notes |
|---------------|-------------------|-------|
| Weather API | 0.5-1s | Fast API |
| Metro Status | 0.3-0.8s | Lightweight JSON |
| Carris Alerts | 1-2s | Larger payload |
| Train Status | 1-3s | Multiple entities |
| GeoJSON Fetch | 2-15s | Depends on dataset size |
| Semantic Search | 0.5-2s | Depends on GPU availability |

### Optimization Strategies

1. **Caching**: Not implemented (real-time data priority)
2. **Retry Logic**: Exponential backoff prevents API overload
3. **Parallel Fetching**: Could be implemented for transport summary
4. **Vector Store**: Persistent storage avoids re-indexing

---

## Integration with LangGraph Agent

All tools are integrated via the `@tool` decorator from LangChain:

```python
from langchain_core.tools import tool

@tool
def example_tool(param: str) -> str:
    """Tool description for LLM."""
    # Implementation
    return result
```

The agent:
1. **Selects tools** based on user query analysis
2. **Calls tools** with appropriate parameters
3. **Formats results** for natural language response
4. **Handles errors** transparently

**See**: [agent_architecture.md](../architecture/agent_architecture.md) for agent design.

---

## Testing Tools

Each tool module includes a `if __name__ == "__main__"` test block:

```bash
# Test IPMA tools
python tools/ipma_api.py

# Test transport tools
python tools/transport_api.py

# Test Dados Abertos
python tools/dados_abertos.py

# Test VisitLisboa (requires vector store)
python tools/visitlisboa_api.py

# Test vector store
python tools/vector_store.py --test
```

---

## Dependencies

**Core**:
- `requests`: HTTP requests
- `langchain-core`: Tool integration
- `pandas`: Data manipulation (Dados Abertos)

**Vector Store**:
- `chromadb`: Vector database
- `langchain-chroma`: ChromaDB integration
- `langchain-huggingface`: Embedding models
- `sentence-transformers`: Embedding generation

**Utilities**:
- `tqdm`: Progress bars
- `logging`: Structured logging

---

## Future Enhancements

### Planned Features

1. **Caching Layer**: Redis for frequent queries
2. **Rate Limiting**: Prevent API overuse
3. **Batch Requests**: Parallel tool calls
4. **Historical Data**: Store past conditions for trends
5. **Location Services**: Geocoding user addresses

### Tool Additions

- **Restaurant Recommendations**: Zomato/TripAdvisor integration
- **Ticket Booking**: Direct booking APIs
- **Navigation**: Google Maps/OSRM routing
- **Crowd Monitoring**: Real-time crowd density

---

*See individual tool documentation for complete API references, examples, and implementation details.*
