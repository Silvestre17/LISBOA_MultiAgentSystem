# 🌐 Data Sources and Schemas

LISBOA combines live requests, short-lived caches, static local data, scheduled source snapshots, and release-backed fallbacks. "Grounded" therefore does not mean that every value is live or retrieved through RAG.

## 🧾 Sources and Freshness

| Source | Runtime Access and Freshness | Main Consumer |
|---|---|---|
| IPMA | Forecasts and warnings requested at runtime, cached for 5 minutes; the provider horizon is five days | `WeatherAgent` |
| Metro de Lisboa | Official API with a public status fallback; responses cached for 60 seconds and station data for 24 hours, with a static fallback | `TransportAgent` |
| Carris Metropolitana | Live REST data; vehicles cached for 30 seconds and stops, lines, and routes for 24 hours, with a stale-cache fallback | `TransportAgent` |
| Carris Urban | GTFS-RT cached for 30 seconds plus static GTFS in SQLite; freshness checks can keep a usable older database | `TransportAgent` |
| CP / Comboios.live | Live status and schedule calls plus CP GTFS in SQLite; conditional static refresh and a one-hour station cache | `TransportAgent` |
| VisitLisboa events | Scheduled JSON snapshot, ChromaDB and structured retrieval, and a direct JSON fallback | `ResearcherAgent`; the Planner through gathered evidence |
| VisitLisboa places | Scheduled JSON snapshot, ChromaDB and structured retrieval, and a direct JSON fallback | `ResearcherAgent`; the Planner through gathered evidence |
| Lisboa Card guide (PDF) | Static April 2024 edition, indexed in `lisbon_pdf` | `ResearcherAgent`; the Planner through gathered evidence |
| Lisboa Aberta | Local metadata snapshot, with GeoJSON datasets fetched on demand | `ResearcherAgent` |
| Location resolution | Local aliases, gazetteer, and station indices, then cached Nominatim and Photon lookups with ambiguity handling | The graph and several tool modules |
| Web knowledge | Constrained Wikipedia, Tavily, and DuckDuckGo fallback for Lisbon history, culture, or very current context | `ResearcherAgent` |

> [!IMPORTANT]
> Transport release assets are last-known-good startup fallbacks. They do not replace the live feeds queried at request time, and "no reported disruption" does not prove that a service is running at the requested time.

## ⏱️ Automation and Delivery

| Workflow | Trigger | Output |
|---|---|---|
| `data_pipeline.yml` | Daily at **04:00 UTC**; manual runs can target events, places, or both | Events refreshed daily and places on Mondays, committed as VisitLisboa JSON artifacts |
| `sync_vector_db.yml` | After a successful `Update Lisbon Data` run; manual runs | Incremental events and places sync, published as complete or staging GitHub Release assets with manifests |
| `sync_transport_runtime_data.yml` | Daily at **03:25 UTC**; pushes that change the transport release code; manual runs | Carris Urban and CP runtime ZIP files and a manifest, replaced in place |
| `deploy_huggingface_space.yml` | Pushes to `main` that change the app; completed vector or transport syncs; manual runs | Deployment bundle uploaded to the Hugging Face Space |

GitHub Actions schedules run in UTC, so 04:00 UTC is 05:00 in Lisbon during summer time. The Lisboa Aberta metadata snapshot is **not** part of the daily workflow: it is refreshed manually with its collection script, and the selected GeoJSON datasets are fetched at runtime.

## 🗂️ VisitLisboa Artifacts

### 🎭 Events

| Item | Value |
|---|---|
| Collector | `data_collection/webscraping/visitlisbon_events.py` |
| Artifact | `data_collection/webscraping/events.json` |
| Top-level fields | `url`, `title`, `category`, `short_description`, `full_description`, `image_urls`, `video_urls`, `dates`, `schedule_notes`, `price`, `venue_name`, `venue_locations`, `location`, `buy_tickets_url`, `information_links` |

### 🏛️ Places

| Item | Value |
|---|---|
| Collector | `data_collection/webscraping/visitlisbon_places.py` |
| Artifact | `data_collection/webscraping/places.json` |
| Top-level fields | `url`, `title`, `category`, `short_description`, `full_description`, `image_urls`, `video_urls`, `features`, `location`, `contact_info`, `social_media`, `schedules`, `tickets_offers`, `tripadvisor`, `information_links`, `additional_sections`, `lisboa_card_benefit`, `lisboa_card_discount`, `lisbon_tourism_member` |

Fields such as `dates`, `venue_locations`, `location`, `contact_info`, `schedules`, `tickets_offers`, and `tripadvisor` can hold nested objects or lists, and some fields are missing in part of the records. Consumers must handle absent values rather than render placeholders.

## 🏥 Lisboa Aberta Metadata

| Item | Value |
|---|---|
| Collector | `data_collection/webscraping/dadosabertos.gov_lisboa.py` and the validation notebook `visitlisbon_dadosabertoslx.ipynb` |
| Runtime snapshot | `data_collection/webscraping/lisbon_datasets_clean.json` |
| Top-level fields | `title`, `description`, `file_formats`, `last_updated`, `stable_url`, `url_portal`; a few unavailable records also carry an explanatory `stable_url_comment` |
| Retrieval | Local metadata ranking, then an on-demand GeoJSON fetch and schema-tolerant feature extraction |

The system deliberately does not embed every municipal dataset in the vector store.

## 🚍 Local Transport Data

| Operator | Local Development Files | Hosted Runtime |
|---|---|---|
| Carris Urban | `data/carris/carris.db`, `data/carris/metadata.json` | Writable runtime directory; live or static initialization first, with the release ZIP as a last-known-good fallback |
| CP | `data/cp/cp_gtfs.db`, `data/cp/metadata.json`, `data/cp/gtfs.zip` | Writable runtime directory; live or static initialization first, with the release ZIP as a last-known-good fallback |

`LISBOA_RUNTIME_DATA_DIR` relocates the generated data. The transport workflow publishes fixed-name assets under the `transport-data-latest` release tag, so the release does not grow over time.

## 🧠 Vector Database

| Item | Value |
|---|---|
| Default local path | `data/vector_db/` |
| Override | `VECTOR_DB_DIR` |
| Hosted path | Writable runtime storage under `LISBOA_RUNTIME_DATA_DIR` |
| Release hydration | The configured `vector_db.zip` is downloaded when no usable local `chroma.sqlite3` exists |
| Embedding model | `BAAI/bge-m3` |
| Collections | `lisbon_pdf`, `lisbon_places`, and `lisbon_events` |

### Sync Semantics

- `lisbon_places` and `lisbon_events` use stable document IDs, content hashes, incremental inserts, updates, and deletions, and resumable `_sync_state` checkpoints.
- `lisbon_pdf` is indexed once when absent and rebuilt explicitly with `--rebuild-pdf` when the source document changes.
- The CI workflow restores the latest release, processes bounded batches, and publishes complete or staging release assets; `data/vector_db/` is never committed.

To inspect the local collections:

```bash
python tools/vector_store.py --stats
```

## 📌 Operational Boundaries

- Metro de Lisboa, Carris Urban, Carris Metropolitana, and CP have distinct operator and geographic scopes; see the [Tools Reference](./03_TOOLS_REFERENCE.md).
- Structured APIs, GTFS/GTFS-RT, geocoding, and GeoJSON are grounded integrations, but not vector retrieval.
- Release-backed data makes startup more resilient; it does not guarantee the current service state.
- `tools/vector_store.py` and `tools/location_resolver.py` support the runtime but are not counted among the 45 exported tools.
