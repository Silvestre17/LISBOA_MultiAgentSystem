# 🌐 Data Sources and Schemas

This page documents what is live, what is periodically refreshed, and what is stored locally for LISBOA.

> [!NOTE]
> Refresh cadences below reflect the GitHub Actions workflows under `.github/workflows/`. Manual workflow dispatches can override the default schedule.

## 🧾 Source Summary

| Source | Type | Access pattern | Refresh model | Main consumers |
|--------|------|----------------|---------------|----------------|
| IPMA | Live API | Direct runtime call | Live on request | `WeatherAgent` |
| Metro de Lisboa | Live API + Public Fallback | Direct runtime call | Live on request | `TransportAgent` |
| Carris Metropolitana | Live REST API | Direct runtime call | Live on request | `TransportAgent` |
| Carris Urban | GTFS + GTFS-RT | Local SQLite + live feed | Live plus cached static support data | `TransportAgent` |
| CP / Comboios.live | Live API + GTFS Support Data | Direct runtime call plus local support files | Live on request | `TransportAgent` |
| VisitLisboa places | Scraped JSON + Vector Store | Local JSON + semantic retrieval | Weekly on Mondays by workflow | `ResearcherAgent`, `PlannerAgent` |
| VisitLisboa events | Scraped JSON + Vector Store | Local JSON + semantic retrieval | Daily by workflow | `ResearcherAgent`, `PlannerAgent` |
| Official Lisbon guide PDF | Static Document + Vector Store | Local file + semantic retrieval | Rebuilt on demand | `ResearcherAgent`, `PlannerAgent` |
| Lisboa Aberta | Open GeoJSON Datasets | Local metadata + on-demand dataset fetch | Metadata refreshed by collection scripts, datasets fetched live | `ResearcherAgent` |
| Web knowledge | Web Search Fallback | Runtime lookup | On request | `ResearcherAgent` |

## ⏱️ Refresh and Staleness Model

Not every source ages the same way. The runtime mixes **live-on-request** queries, **scheduled repository snapshots**, and **local support stores**.

### Live on Request

Queried at runtime, not versioned as scraped repository snapshots:
IPMA · Metro de Lisboa · Carris Metropolitana · Carris GTFS-RT · Comboios.live · Lisboa Aberta dataset contents (when a specific dataset is fetched on demand).

### Scheduled Repository Refresh

| Workflow | Schedule or trigger | What it updates |
|----------|---------------------|-----------------|
| `data_pipeline.yml` | daily at **04:00 UTC**, plus manual selector | VisitLisboa events daily, places weekly on Mondays; manual runs target events, places, or both |
| `sync_vector_db.yml` | `workflow_run` after a successful data update, plus manual trigger | incremental ChromaDB sync for changed collections |

Vector collections are updated **incrementally** rather than rebuilt from scratch.

## 🗂️ Scraped JSON Artefacts

### 🎭 *VisitLisboa* Events

| Item | Value |
|------|-------|
| Script | `data_collection/webscraping/visitlisbon_events.py` |
| Output | `data_collection/webscraping/events.json` |
| Used by | vector sync, `ResearcherAgent`, `PlannerAgent` |

Common fields include:

- `url`
- `title`
- `category`
- `short_description`
- `full_description`
- `image_urls`
- `video_urls`
- `dates`
- `price`
- `venue_name`
- `location`
- `buy_tickets_url`
- `information_links`

### 🏛️ *VisitLisboa* Places

| Item | Value |
|------|-------|
| Script | `data_collection/webscraping/visitlisbon_places.py` |
| Output | `data_collection/webscraping/places.json` |
| Used by | vector sync, `ResearcherAgent`, `PlannerAgent` |

Common fields include:

- `url`
- `title`
- `category`
- `short_description`
- `full_description`
- `image_urls`
- `video_urls`
- `features`
- `location`
- `contact_info`
- `social_media`
- `schedules`
- `tickets_offers`
- `tripadvisor`

## 🏥 *Lisboa Aberta* Metadata Layer

| Item | Value |
|------|-------|
| Metadata file | `data_collection/webscraping/lisbon_datasets_clean.json` |
| Retrieval model | structured local metadata + on-demand GeoJSON fetch |
| Used by | dataset discovery, category browsing, keyword search, detail lookup |

The system does **not** embed every Lisboa Aberta dataset into the vector store. Instead, it keeps metadata locally and fetches relevant datasets on demand.

## 🚍 Local Transport Support Data

These artefacts support faster local lookups and reduce repeated parsing of static transport files.

| Layer | Local artefacts | Purpose |
|-------|------------------|---------|
| Carris Urban | `data/carris/carris.db`, `data/carris/metadata.json` | Runtime stop, Route, and GTFS support |
| CP | `data/cp/cp_gtfs.db`, `data/cp/metadata.json`, `data/cp/gtfs.zip` | Local schedule support and Reproducible reference data |

## 🧠 Vector Database

### Storage and Collections

| Item | Value |
|------|-------|
| Storage directory | `data/vector_db/` |
| Embedding model | `BAAI/bge-m3` |
| Collections | `lisbon_pdf`, `lisbon_places`, `lisbon_events` |
| Language support | Multilingual Retrieval, with Portuguese and English coverage in the indexed material |

The vector store supports multilingual retrieval, but the runtime emits final user-facing answers only in PT-PT or English.

### Sync Semantics

The vector-store update flow is incremental:

- Documents receive stable identifiers.
- Metadata stores a SHA-256 content hash.
- New content is inserted; changed content is updated; removed content is deleted from the affected collection.

This allows the GitHub Actions sync workflow to process updates in batches instead of rebuilding the full store every time.

To inspect the current collection state locally:

```bash
python tools/vector_store.py --stats
```

## 📌 Operational Boundaries

- Exported runtime tools are counted from `tools/__init__.py`.
- `tools/vector_store.py` is operational infrastructure, not an exported runtime tool.
- VisitLisboa semantic retrieval depends on both local JSON artefacts and the vector store.
- Lisboa Aberta service discovery is intentionally handled through structured on-demand fetches rather than bulk embedding.
