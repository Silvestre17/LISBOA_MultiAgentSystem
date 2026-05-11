# 🛠️ LISBOA Tools Reference

The authoritative exported tool registry is [`tools/__init__.py`](../tools/__init__.py), which currently exposes **45 LangChain tools** used by the runtime.

> [!IMPORTANT]
> `tools/vector_store.py` is operational support and CLI infrastructure. It matters for the system, but it is **not** counted as one of the 45 exported runtime tools.

## 📦 Inventory by Domain

| Domain | Module | Count | Coverage |
|--------|--------|------:|----------|
| Weather | `tools/ipma_api.py` | 4 | Warnings, Forecast, Current Summary, Portugal-Wide Overview |
| Metro de Lisboa | `tools/metrolisboa_api.py` | 6 | Line Status, Wait Times, Frequencies, Station Discovery |
| Carris Metropolitana | `tools/carrismetropolitana_api.py` | 8 | Alerts, Stops, Lines, Routes, Live Bus Positions, Departures |
| Carris Urban | `tools/carris_api.py` | 8 | Stops, Routes, Departures, Arrivals, ETA, Frequency, Realtime Vehicles |
| CP trains | `tools/cp_api.py` | 6 | Station Search, Schedules, Routes, Trip Planning, Frequency, Status |
| Multimodal transport | `tools/transport_api.py` | 2 | Combined Network Status and Route Planning |
| Lisboa Aberta open data | `tools/dados_abertos.py` | 5 | Nearby Services, Dataset Discovery, Category Browsing |
| VisitLisboa semantic search | `tools/visitlisboa_api.py` | 5 | Events, Places, Categories, Tourism Knowledge Search |
| Web knowledge | `tools/web_knowledge.py` | 1 | History and Culture Fallback Search |
| **Total exported tools** |  | **45** |  |

## 🤖 Inventory by Runtime Agent

| Agent | Assigned tools | Composition |
|------|---------------:|-------------|
| `WeatherAgent` | 4 | IPMA only |
| `TransportAgent` | 30 | Metro 6 + Carris Metropolitana 8 + Carris Urban 8 + CP 6 + multimodal 2 |
| `ResearcherAgent` | 11 | VisitLisboa 5 + Lisboa Aberta 5 + web 1 |
| `SupervisorAgent` | 0 | routing only |
| `QualityAssuranceAgent` | 0 | validation only |
| `PlannerAgent` | 0 | synthesis only |

## 🔍 Detailed Inventory

### 🌦️ Weather, 4 Tools

| Tool | Purpose |
|------|---------|
| `get_weather_warnings` | Retrieve active meteorological warnings |
| `get_weather_forecast` | Retrieve a focused Lisbon forecast window, with `days` and optional `day_offset` within the 5-day IPMA horizon |
| `get_current_weather_summary` | Summarize current conditions for Lisbon |
| `get_portugal_weather_overview` | Compare weather across Portugal locations |

### 🚇 Metro de Lisboa, 6 Tools

| Tool | Purpose |
|------|---------|
| `get_metro_status` | Retrieve current line status |
| `get_metro_wait_time` | Retrieve station-level wait times |
| `get_metro_line_wait_times` | Retrieve wait times across a full line |
| `find_nearest_metro` | Find the nearest metro station from coordinates |
| `get_metro_frequency` | Retrieve train frequency schedules |
| `get_all_metro_stations` | List all metro stations |

### 🚌 Carris Metropolitana, 8 Tools

| Tool | Purpose |
|------|---------|
| `get_carris_metropolitana_alerts` | List active service alerts |
| `get_carris_metropolitana_stop_info` | Inspect stop metadata |
| `search_carris_metropolitana_lines` | Search line information |
| `find_bus_routes` | Discover bus routes between locations |
| `get_real_time_bus_positions` | Inspect live bus positions with optional filtering |
| `get_bus_realtime_locations` | Retrieve real-time GPS bus locations |
| `get_bus_next_departures` | Retrieve upcoming departures or route stop information |
| `find_direct_bus_lines` | Find direct bus connections |

### 🚋 Carris Urban, 8 Tools

| Tool | Purpose |
|------|---------|
| `carris_get_stops` | Search and inspect Carris stops |
| `carris_get_routes` | Retrieve route details |
| `carris_get_next_departures` | Retrieve next departures at a stop |
| `carris_find_routes_between` | Find routes between stops |
| `carris_get_realtime_vehicles` | Track live vehicles |
| `carris_get_arrivals` | Retrieve arrivals at a stop |
| `carris_vehicle_eta` | Estimate vehicle arrival time at a stop |
| `carris_get_service_frequency` | Inspect service frequency and headway |

### 🚆 CP Trains, 6 Tools

| Tool | Purpose |
|------|---------|
| `get_train_status` | Retrieve train status and delays |
| `search_cp_stations` | Search CP stations in the supported network |
| `get_train_schedule` | Retrieve schedule departures |
| `get_cp_routes` | Inspect train routes and lines |
| `plan_train_trip` | Plan a train trip between stations |
| `get_train_frequency` | Inspect service frequency and headway |

### 🔀 Multimodal Transport, 2 Tools

| Tool | Purpose |
|------|---------|
| `get_transport_summary` | Summarize operational status across transport modes |
| `get_route_between_stations` | Plan multimodal routes across providers |

### 🏥 Lisboa Aberta, 5 Tools

| Tool | Purpose |
|------|---------|
| `find_nearby_services` | Search nearby services by category and distance |
| `list_available_datasets` | List available Lisboa Aberta datasets |
| `get_dataset_details` | Inspect dataset metadata |
| `find_place_in_datasets` | Search place names across datasets |
| `list_service_categories` | Browse service-category groupings |

### 🏛️ VisitLisboa Semantic Retrieval, 5 Tools

| Tool | Purpose |
|------|---------|
| `search_cultural_events` | Semantic search for cultural events |
| `search_places_attractions` | Semantic search for places and attractions |
| `get_event_categories` | List supported event categories |
| `get_place_categories` | List supported place categories |
| `search_lisbon_knowledge` | General semantic tourism-knowledge search |

### 🌍 Web Knowledge, 1 Tool

| Tool | Purpose |
|------|---------|
| `search_history_culture` | Fallback web search for Lisbon history and culture |

## 🔌 Upstream APIs and Feeds Behind the Tools

The tool layer integrates with the following source families:

- **IPMA** open-data endpoints
- **Metro de Lisboa** official API plus public fallback status endpoint
- **Carris Metropolitana** REST API
- **Carris Urban** GTFS and GTFS-RT feeds
- **Comboios.live** plus local **CP GTFS** support data
- **Lisboa Aberta** GeoJSON datasets
- **VisitLisboa** scraped JSON plus ChromaDB retrieval

## 🛡️ Reliability Patterns in the Tool Layer

- Readable failure messages instead of hard crashes at tool level.
- Targeted retries and caching for network-heavy sources; public fallback endpoints where available.
- Local reference stores for Carris and CP support workflows.
- Transport answers rebuild one canonical localized source footer from the operators actually invoked, collapsing duplicate footers and avoiding citation of operators that were not used.

## 🧠 Vector-Store CLI Support

`tools/vector_store.py` supports the following operational flags:

| Flag | Purpose |
|------|---------|
| `--rebuild-all` | force a full rebuild of all collections |
| `--rebuild-pdf` | rebuild only the PDF collection |
| `--rebuild-places` | rebuild only the places collection |
| `--rebuild-events` | rebuild only the events collection |
| `--test` | run search-oriented smoke checks |
| `--stats` | show collection statistics |
| `--no-gpu` | force CPU-only execution |
| `--max-docs` | limit the number of documents processed in one pass |

Example commands:

```bash
python tools/vector_store.py
python tools/vector_store.py --stats
python tools/vector_store.py --test
python tools/vector_store.py --no-gpu --max-docs 200
```

## ✅ Local Smoke Checks

```bash
python tools/ipma_api.py
python tools/transport_api.py
python tools/dados_abertos.py
python tools/visitlisboa_api.py
python tools/vector_store.py --test
```
