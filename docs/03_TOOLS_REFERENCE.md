# Tools

This document describes the tool layer used by the Lisbon Urban Assistant.

The authoritative exported tool list is defined in `tools/__init__.py` and contains **42 tools**.

Tools are implemented as LangChain tools (via `@tool`) and typically return a formatted Markdown string.

Local tool invocation pattern (for debugging):

```python
result = get_weather_forecast.invoke({"days": 3})
print(result)
```

## Table of contents

1. Tool inventory (42)
2. Weather (IPMA)
3. Transport (Metro, Carris Metropolitana, CP, Carris Urban)
4. Open data (Lisboa Aberta)
5. VisitLisboa semantic search
6. Web knowledge
7. Vector store (internal API and CLI)
8. Error handling and reliability
9. Testing tools locally

## 1. Tool inventory (42)

GTFS-RT stop status codes (as used by the underlying feed):

- `INCOMING_AT` (0): vehicle is approaching the stop
- `STOPPED_AT` (1): vehicle is currently at the stop
- `IN_TRANSIT_TO` (2): vehicle is traveling toward the stop

Tram line quick reference (common tourist lines):

- `12E`: Martim Moniz to Praca Luis de Camoes
- `15E`: Praca da Figueira to Alges (Belem corridor)
- `18E`: Cais do Sodre to Cemiterio da Ajuda
- `24E`: Praca Luis de Camoes to Campolide
- `25E`: Praca da Figueira to Campo de Ourique
- `28E`: Martim Moniz to Praca Luis de Camoes (Graca, Alfama, Chiado)

### Weather (IPMA), 4

- `get_weather_warnings(area="LSB")`
- `get_weather_forecast(days=3)`
- `get_current_weather_summary()`
- `get_portugal_weather_overview(day=0)`
### Metro de Lisboa, 6

- `get_metro_status()`
- `get_metro_wait_time(station)`
- `get_metro_line_wait_times(line)`
- `find_nearest_metro(latitude=None, longitude=None, near_location_name=None)`
- `get_metro_frequency(line, day_type="weekday")`
- `get_all_metro_stations()`

### Carris Metropolitana (AML buses), 8

- `get_real_time_bus_positions(line_id=None, location=None, radius_km=1.0)`

- `get_carris_metropolitana_alerts()`
- `get_carris_metropolitana_stop_info(stop_id)`
- `search_carris_metropolitana_lines(query)`
- `find_bus_routes(origin, destination)`
- `get_bus_realtime_locations(line_id=None)`
- `get_bus_next_departures(line_id, stop_id="", start_time="")`
- `find_direct_bus_lines(origin, destination)`

### CP (trains), 5

- `get_train_status()`
- `search_cp_stations(query)`
- `get_train_schedule(station_name, limit=10)`
- `get_cp_routes()`
- `plan_train_trip(origin, destination)`

### Multimodal transport, 2

- `get_transport_summary()`
- `get_route_between_stations(origin, destination)`

### Lisboa Aberta open data, 4

- `find_nearby_services(service_type, user_lat=None, user_lon=None, near_location_name=None, max_results=5)`
- `list_available_datasets(category=None)`
- `get_dataset_details(dataset_name)`
- `find_place_in_datasets(query, max_results=5)`

### VisitLisboa semantic search, 5

- `search_cultural_events(query=None, category=None, date_filter=None, max_results=10)`
- `search_places_attractions(query=None, category=None, max_results=10)`
- `get_event_categories()`
- `get_place_categories()`
- `search_lisbon_knowledge(query, max_results=10)`

### Carris Urban (Lisbon city), 7

- `carris_get_stops(query="", limit=None)`
- `carris_get_routes(route_type="", route_id="", limit=50)`
- `carris_get_next_departures(stop_id, limit=10)`
- `carris_find_routes_between(origin, destination, search_radius_km=0.4)`
- `carris_get_realtime_vehicles(route_id="", vehicle_type="")`
- `carris_get_arrivals(stop_id, limit=10)`
- `carris_vehicle_eta(route_short_name, stop_name)`

### Web knowledge, 1

- `search_history_culture(query, language="pt")`

## 2. Weather (IPMA)

Module: `tools/ipma_api.py`

Key endpoints:

- Warnings: `https://api.ipma.pt/open-data/forecast/warnings/warnings_www.json`
- Daily city forecast: `https://api.ipma.pt/open-data/forecast/meteorology/cities/daily/{globalIdLocal}.json`

Project defaults:

- Lisbon `globalIdLocal`: `1110600`
- Warning area code: `LSB`

Typical usage patterns:

- Use `get_current_weather_summary()` when the user asks, "Do I need an umbrella today?"
- Use `get_weather_warnings()` when the user asks about storms, heatwaves, or risk conditions.

## 3. Transport

This project separates:

- Carris Urban (Lisbon city buses and trams)
- Carris Metropolitana (AML suburban buses)

### 3.1 Metro de Lisboa

Module: `tools/metrolisboa_api.py`

Data sources:

- Official API with OAuth2 (requires credentials)
- Public fallback status endpoint: `https://app.metrolisboa.pt/status/getLinhas.php`

Environment variables:

- `METRO_CONSUMER_KEY`
- `METRO_CONSUMER_SECRET`

### 3.2 Carris Metropolitana (AML)

Module: `tools/carrismetropolitana_api.py`

Base API: `https://api.carrismetropolitana.pt/v2/`

Important response fields:

- Alerts: check `active_period` and `description_text`.

### 3.3 CP (Comboios de Portugal)

Module: `tools/cp_api.py`

Real-time API:

- Stations: `https://comboios.live/api/stations`
- Vehicles: `https://comboios.live/api/vehicles`

The vehicles feed includes fields such as `trainNumber`, `delay`, and `status`.

### 3.4 Carris Urban (Lisbon city)

Module: `tools/carris_api.py`

Data sources:

- GTFS static: `https://gateway.carris.pt/gateway/gtfs/api/v2.8/GTFS`
- GTFS-RT vehicle positions (Protocol Buffers): `https://gateway.carris.pt/gateway/gtfs/api/v2.8/GTFS/realtime/vehiclepositions`

Local storage:

- SQLite: `data/carris/carris.db`
- Metadata: `data/carris/metadata.json`

Recommended workflows:

1. "When is the next bus or tram at stop X?"
   - `carris_get_stops("stop name")` to find `stop_id`
   - `carris_get_arrivals(stop_id)` for best real-time arrivals

2. "When will route 28E arrive at stop Y?"
   - `carris_vehicle_eta("28E", "stop name")`

Implementation notes:

- Static GTFS is downloaded as a ZIP and converted into SQLite for fast querying.
- Real-time vehicle positions are consumed from GTFS-RT (Protocol Buffers).
- For most user questions about "next vehicle at a stop", prefer `carris_get_arrivals(...)` (it fuses schedule plus real-time when available).

## 4. Open data (Lisboa Aberta)

Module: `tools/dados_abertos.py`

Dataset discovery:

- `https://dados.cm-lisboa.pt/dataset?res_format=GeoJSON&_tags_limit=0`

Highlights:

- On-demand GeoJSON fetching.
- Proximity filtering with distance calculations.
- Optional geocoding by place name: `find_nearby_services(..., near_location_name="Rossio")` (tries Open Data first, then a Nominatim fallback).

Example:

```python
places = find_place_in_datasets.invoke({"query": "Fernando Pessoa", "max_results": 3})
```

## 5. VisitLisboa semantic search

Module: `tools/visitlisboa_api.py`

Data inputs:

- Events JSON: `data_collection/webscraping/events.json`
- Places JSON: `data_collection/webscraping/places.json`

Search behavior:

- Semantic search via the Chroma vector database when available.
- Date parsing and filtering for events.
- Fallback to JSON keyword search if the vector store is not available.

## 6. Web knowledge

Module: `tools/web_knowledge.py`

Fallback order:

1. Tavily (requires `TAVILY_API_KEY`)
2. DuckDuckGo
3. Wikipedia

Tool:

- `search_history_culture(query, language="pt")`

Notes:

- Wikipedia language defaults to Portuguese.

## 7. Vector store (internal API and CLI)

Module: `tools/vector_store.py`

Important: the vector store is not exported as a LangChain tool in `tools/__init__.py`. It is an internal service used by other tools (for example VisitLisboa semantic search) and by workflows.

### Persistent storage

- Directory: `data/vector_db/`

### Embeddings

- Model: `BAAI/bge-m3` (configured in `config.py`)

### Programmatic API

`VectorStore` provides:

- `sync_all(...)`
- `get_stats()`
- `search(query, k=5, collections=None, min_score=None)`
- `search_with_scores(query, k=5, collections=None)`

Score notes:

- Chroma returns L2 distance (lower is better).

### CLI usage

```bash
python tools/vector_store.py
```

Batch mode (used by GitHub Actions):

```bash
python tools/vector_store.py --no-gpu --max-docs 200
```

Exit codes:

- `0`: complete
- `2`: more work pending

Collections:

- `lisbon_pdf`: static PDF guide (indexed once)
- `lisbon_places`: VisitLisboa places
- `lisbon_events`: VisitLisboa events

Incremental sync approach:

- Each document is assigned a stable ID derived from `(source, url)`.
- A SHA-256 content hash is stored in metadata to detect changes.
- Sync removes deleted items from collections.

GitHub Actions safety:

- The script handles SIGTERM and exits with code `2` after completing the current batch, so partial progress can still be committed.
- Telemetry is disabled via environment variables inside the script.

## 8. Error handling and reliability

Common failure cases and what users see:

- API unavailable: tools return a readable status message and do not crash the agent.
- Timeout: tools use per-call timeouts and fail fast instead of hanging.
- No results: tools return an explicit "no matches" message.

Source-specific fallbacks:

- Metro: falls back to the public status endpoint when official credentials are missing.
- Web knowledge: waterfall strategy (Tavily, DuckDuckGo, Wikipedia).

## 9. Performance defaults (timeouts, retries, caches)

This section documents current performance defaults as implemented in the tool modules.

Weather (IPMA), `tools/ipma_api.py`:

- Request timeout: 10 seconds.
- Cache: 5 minute TTL for weather API responses.

Metro de Lisboa, `tools/metrolisboa_api.py`:

- Request timeout: 15 seconds.
- Retries: 3 attempts with exponential backoff factor 2.
- Cache: 60 second TTL for real time transport JSON.
- Station list cache: 24 hour expiration (station metadata changes rarely).

Carris Metropolitana (AML), `tools/carrismetropolitana_api.py`:

- Request timeout: 15 seconds.
- Cache: 30 second TTL for real time vehicle locations.

Carris Urban (Lisbon city), `tools/carris_api.py`:

- GTFS static download timeout: 120 seconds.
- Real time request timeout: 15 seconds.
- GTFS RT cache: 30 seconds.

CP trains, `tools/cp_api.py`:

- Typical request timeout (comboios.live): 15 seconds.
- GTFS download uses a longer timeout for the ZIP download step.

Lisboa Aberta open data, `tools/dados_abertos.py`:

- Request timeout: 15 seconds.
- Retries: 3 attempts with exponential backoff (2s, 4s, 8s).

## 10. Testing tools locally

Quick checks:

```bash
python tools/ipma_api.py
python tools/transport_api.py
python tools/dados_abertos.py
python tools/visitlisboa_api.py
python tools/vector_store.py --test
```

## References

Carris. (n.d.). GTFS data gateway. https://gateway.carris.pt/
Carris Metropolitana. (n.d.). Carris Metropolitana API. https://api.carrismetropolitana.pt/
Comboios.live. (n.d.). Comboios.live API. https://comboios.live/
Instituto Portugues do Mar e da Atmosfera. (n.d.). IPMA Open Data API. https://api.ipma.pt/
Metro de Lisboa. (n.d.). Metro de Lisboa API Store. https://api.metrolisboa.pt/store/
