# 🛠️ LISBOA Tools Reference

The authoritative tool registry is [`tools/__init__.py`](../tools/__init__.py), which exports **45 LangChain tools** used by the runtime.

> [!IMPORTANT]
> `tools/vector_store.py` provides operational and CLI support. The system depends on it, but it is **not** one of the 45 exported tools.

## 📦 Inventory by Domain

| Domain | Module | Tools | Coverage |
|---|---|---:|---|
| Weather | `tools/ipma_api.py` | 4 | Warnings, daily forecast, today's summary, and a Portugal-wide overview |
| Metro de Lisboa | `tools/metrolisboa_api.py` | 6 | Line status, wait times, frequencies, and station discovery |
| Carris Metropolitana | `tools/carrismetropolitana_api.py` | 8 | Alerts, stops, lines, routes, live bus positions, and departures |
| Carris Urban | `tools/carris_api.py` | 8 | Stops, routes, departures, arrivals, arrival estimates, frequency, and real-time vehicles |
| CP suburban rail | `tools/cp_api.py` | 6 | Station search, schedules, supported routes, trip planning, frequency, and status |
| Multimodal transport | `tools/transport_api.py` | 2 | Network status and route planning across the supported operators |
| Lisboa Aberta open data | `tools/dados_abertos.py` | 5 | Nearby services, dataset discovery, and category browsing |
| VisitLisboa hybrid retrieval | `tools/visitlisboa_api.py` | 5 | Semantic search, structured filtering, categories, and a JSON fallback |
| Web knowledge | `tools/web_knowledge.py` | 1 | Constrained fallback for Lisbon history, culture, and very current context |
| **Total** | | **45** | |

## 🤖 Inventory by Agent

| Agent | Tools | Composition |
|---|---:|---|
| `WeatherAgent` | 4 | IPMA only |
| `TransportAgent` | 30 | Metro de Lisboa 6, Carris Metropolitana 8, Carris Urban 8, CP 6, and multimodal 2 |
| `ResearcherAgent` | 11 | VisitLisboa 5, Lisboa Aberta 5, and web 1 |
| `SupervisorAgent` | 0 | Routing only |
| `QualityAssuranceAgent` | 0 | Validation only |
| `PlannerAgent` | 0 | Synthesis only |

## 🔍 Detailed Inventory

### 🌦️ Weather (4 Tools)

| Tool | Purpose |
|---|---|
| `get_weather_warnings` | Retrieve active meteorological warnings |
| `get_weather_forecast` | Retrieve a Lisbon forecast window, with `days` and an optional `day_offset` within the five-day IPMA horizon |
| `get_current_weather_summary` | Summarize today's IPMA forecast and active Lisbon warnings |
| `get_portugal_weather_overview` | Compare the weather across locations in Portugal |

### 🚇 Metro de Lisboa (6 Tools)

| Tool | Purpose |
|---|---|
| `get_metro_status` | Retrieve the current line status |
| `get_metro_wait_time` | Retrieve wait times at a station |
| `get_metro_line_wait_times` | Retrieve wait times along a whole line |
| `find_nearest_metro` | Find the nearest station to given coordinates |
| `get_metro_frequency` | Retrieve train frequency schedules |
| `get_all_metro_stations` | List all stations |

### 🚌 Carris Metropolitana (8 Tools)

| Tool | Purpose |
|---|---|
| `get_carris_metropolitana_alerts` | List active service alerts |
| `get_carris_metropolitana_stop_info` | Inspect stop metadata |
| `search_carris_metropolitana_lines` | Search line information |
| `find_bus_routes` | Find bus routes between locations |
| `get_real_time_bus_positions` | Inspect live bus positions, with optional filters |
| `get_bus_realtime_locations` | Retrieve real-time GPS bus locations |
| `get_bus_next_departures` | Retrieve upcoming departures or the stops on a route |
| `find_direct_bus_lines` | Find direct bus connections |

### 🚋 Carris Urban (8 Tools)

| Tool | Purpose |
|---|---|
| `carris_get_stops` | Search and inspect Carris stops |
| `carris_get_routes` | Retrieve route details |
| `carris_get_next_departures` | Retrieve the next departures at a stop |
| `carris_find_routes_between` | Find routes between two stops; answers rank them door to door (walk, wait for the next reachable departure, ride, walk) and show departures, delays, and arrival |
| `carris_get_realtime_vehicles` | Track vehicles in real time |
| `carris_get_arrivals` | Retrieve arrivals at a stop |
| `carris_vehicle_eta` | Estimate when a vehicle will reach a stop |
| `carris_get_service_frequency` | Inspect service frequency and headways |

### 🚆 CP Suburban Rail (6 Tools)

| Tool | Purpose |
|---|---|
| `get_train_status` | Retrieve train status and delays |
| `search_cp_stations` | Search stations in the supported network |
| `get_train_schedule` | Retrieve scheduled departures |
| `get_cp_routes` | Inspect train routes and lines |
| `plan_train_trip` | Plan a train trip between stations, with arrival times and the live delay status |
| `get_train_frequency` | Inspect service frequency and headways |

### 🔀 Multimodal Transport (2 Tools)

| Tool | Purpose |
|---|---|
| `get_transport_summary` | Summarize the operational status across transport modes |
| `get_route_between_stations` | Plan multimodal routes across the supported operators; a Metro leg that rides a section stopped in the live status is marked, and answers then avoid it or recommend another mode |

### 🏥 Lisboa Aberta (5 Tools)

| Tool | Purpose |
|---|---|
| `find_nearby_services` | Search nearby services by category and distance |
| `list_available_datasets` | List the available Lisboa Aberta datasets |
| `get_dataset_details` | Inspect dataset metadata |
| `find_place_in_datasets` | Search place names across datasets |
| `list_service_categories` | Browse service categories |

### 🏛️ VisitLisboa Hybrid Retrieval (5 Tools)

| Tool | Purpose |
|---|---|
| `search_cultural_events` | Search events through semantic retrieval, date and category filters, and a JSON fallback |
| `search_places_attractions` | Search places through semantic retrieval, structured ranking and filters, and a JSON fallback; exclusions ("nada de fado") are removed from the query and applied to titles, categories, and features, and ranked requests weight the rating by the number of reviews |
| `get_event_categories` | List event categories from the local VisitLisboa artifact |
| `get_place_categories` | List place categories from the local VisitLisboa artifact |
| `search_lisbon_knowledge` | Search the indexed Lisboa Card guide and tourism knowledge |

### 🌍 Web Knowledge (1 Tool)

| Tool | Purpose |
|---|---|
| `search_history_culture` | Constrained fallback for Lisbon history, culture, and very current context, through Wikipedia, Tavily, and DuckDuckGo |

## 🔌 Upstream Sources

The tool layer draws on the following sources:

- **IPMA** open-data endpoints
- **Metro de Lisboa** official API, with a public status endpoint as fallback
- **Carris Metropolitana** REST API
- **Carris Urban** GTFS and GTFS-RT feeds
- **Comboios.live**, with local **CP GTFS** support data
- **Lisboa Aberta** GeoJSON datasets
- **VisitLisboa** scraped JSON, with ChromaDB hybrid retrieval
- **Location resolution** through a local gazetteer and aliases, Metro and CP station indices, Nominatim, and Photon; Nominatim reverse geocoding also gives a street address to places grounded from Wikipedia
- **Web fallback** through Wikipedia, Tavily, and DuckDuckGo; operator-specific tools remain the primary source for transport and weather. Inside the AML, Wikipedia pages also ground named places and stop types that VisitLisboa does not list, and are cited as such

`tools/location_resolver.py` is shared support infrastructure used by the graph and several tool modules. It handles the AML scope, ambiguous place names, geocoding fallbacks, and transport-node enrichment, but it is not an exported tool.

## 🚧 Coverage Boundaries

- **Metro de Lisboa** covers the four Metro lines.
- **Carris Urban** covers Lisbon city buses and trams; **Carris Metropolitana** covers AML intermunicipal buses.
- **CP** is limited to the supported AML suburban lines: Cascais, Sintra, Azambuja, and Sado.
- Multimodal routing combines only the operators and modes implemented in the repository.
- Long-distance rail, live Fertagus detail, ferries, ride-hailing, shared bikes and scooters, booking, and ticket purchase are not implemented.

## 🛡️ Reliability Patterns

- Readable failure messages instead of crashes at the tool level
- Targeted retries and caching for network-heavy sources, and public fallback endpoints where available
- Local reference stores for the Carris Urban and CP workflows
- One canonical, localized source footer per transport answer, built from the operators actually used, with duplicates collapsed

## 🧠 Vector-Store CLI

`tools/vector_store.py` accepts the following flags:

| Flag | Purpose |
|---|---|
| `--rebuild-all` | Rebuild all collections |
| `--rebuild-pdf` | Rebuild only the Lisboa Card guide collection (`lisbon_pdf`) |
| `--rebuild-places` | Rebuild only the places collection |
| `--rebuild-events` | Rebuild only the events collection |
| `--test` | Run search smoke checks |
| `--stats` | Show collection statistics |
| `--no-gpu` | Force CPU-only execution |
| `--max-docs` | Limit the documents processed per JSON collection in one run |

```bash
python tools/vector_store.py
python tools/vector_store.py --stats
python tools/vector_store.py --test
python tools/vector_store.py --no-gpu --max-docs 200
```

## ✅ Local Smoke Checks

Each module below can be run directly to check its integration:

```bash
python tools/ipma_api.py
python tools/transport_api.py
python tools/dados_abertos.py
python tools/visitlisboa_api.py
python tools/vector_store.py --test
```
