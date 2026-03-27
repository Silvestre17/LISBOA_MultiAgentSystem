# 🌐 Data Sources and Schemas

This page documents what is live, what is periodically refreshed, and what is stored locally for LISBOA.

## 🧾 Source Summary

| Source | Type | Access pattern | Refresh model | Main consumers |
|--------|------|----------------|---------------|----------------|
| IPMA | live API | direct runtime call | live on request | `WeatherAgent` |
| Metro de Lisboa | live API + public fallback | direct runtime call | live on request | `TransportAgent` |
| Carris Metropolitana | live REST API | direct runtime call | live on request | `TransportAgent` |
| Carris Urban | GTFS + GTFS-RT | local SQLite + live feed | live plus cached static support data | `TransportAgent` |
| CP / Comboios.live | live API + GTFS support data | direct runtime call plus local support files | live on request | `TransportAgent` |
| VisitLisboa places | scraped JSON + vector store | local JSON + semantic retrieval | weekly on Mondays by workflow | `ResearcherAgent`, `PlannerAgent` |
| VisitLisboa events | scraped JSON + vector store | local JSON + semantic retrieval | daily by workflow | `ResearcherAgent`, `PlannerAgent` |
| Official Lisbon guide PDF | static document + vector store | local file + semantic retrieval | rebuilt on demand | `ResearcherAgent`, `PlannerAgent` |
| Lisboa Aberta | open GeoJSON datasets | local metadata + on-demand dataset fetch | metadata refreshed by collection scripts, datasets fetched live | `ResearcherAgent` |
| Web knowledge | web search fallback | runtime lookup | on request | `ResearcherAgent` |

## ⏱️ Refresh and Staleness Model

> [!NOTE]
> Not all LISBOA data layers age in the same way. Some are live on request, some are refreshed daily or weekly into repository artefacts, and some are generated locally to speed up transport support workflows.

### Live on Request

These sources are queried at runtime and are not versioned as scraped repository snapshots:

- IPMA
- Metro de Lisboa
- Carris Metropolitana
- Carris GTFS-RT
- Comboios.live
- Lisboa Aberta dataset contents when a specific dataset is fetched on demand

### Scheduled Repository Refresh

| Workflow | Schedule or trigger | What it updates |
|----------|---------------------|-----------------|
| `data_pipeline.yml` | daily at **04:00 UTC**, plus manual selector | VisitLisboa event JSON every day and place JSON on Mondays; manual runs can target events, places, or both |
| `sync_vector_db.yml` | `workflow_run` after successful data update, plus manual trigger | incremental ChromaDB sync for changed collections |

In practice:

- VisitLisboa **events** are scraped daily
- VisitLisboa **places** are scraped weekly on Mondays during scheduled runs
- manual runs can choose **events**, **places**, or **both**
- vector collections are then updated incrementally instead of being rebuilt from scratch every time

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
| Carris Urban | `data/carris/carris.db`, `data/carris/metadata.json` | runtime stop, route, and GTFS support |
| CP | `data/cp/cp_gtfs.db`, `data/cp/metadata.json`, `data/cp/gtfs.zip` | local schedule support and reproducible reference data |

## 🧠 Vector Database

### Storage and Collections

| Item | Value |
|------|-------|
| Storage directory | `data/vector_db/` |
| Embedding model | `BAAI/bge-m3` |
| Collections | `lisbon_pdf`, `lisbon_places`, `lisbon_events` |
| Language support | multilingual, with Portuguese and English retrieval |

### Sync Semantics

The vector-store update flow is incremental:

- documents receive stable identifiers
- metadata stores a SHA-256 content hash
- new content is inserted
- changed content is updated
- removed content is deleted from the affected collection

This allows the GitHub Actions sync workflow to process updates in batches instead of rebuilding the full store every time.

To inspect the current collection state locally:

```bash
python tools/vector_store.py --stats
```

## 📌 Operational Boundaries

- Exported runtime tools are counted from `tools/__init__.py`
- `tools/vector_store.py` is operational infrastructure, not an exported runtime tool
- VisitLisboa semantic retrieval depends on both local JSON artefacts and the vector store
- Lisboa Aberta service discovery is intentionally handled through structured on-demand fetches rather than bulk embedding
