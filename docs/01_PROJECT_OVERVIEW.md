# 📍 LISBOA Project Overview

LISBOA is a Master's thesis project for the Lisbon Metropolitan Area that combines multi-agent orchestration, live public APIs, municipal open data, and semantic retrieval. The documented public entrypoint is `app.py`.

## 👥 Who LISBOA Serves

| Audience | Typical questions | Main data layers |
|----------|-------------------|------------------|
| **🧳 Tourists** | itineraries, museums, attractions, weather, events, transport between landmarks | VisitLisboa, IPMA, Metro, Carris, CP, multimodal routing |
| **🏠 Residents** | daily mobility, nearby services, local events, urban information | Lisboa Aberta, Metro, Carris Metropolitana, Carris Urban, CP, IPMA |

## 🎓 Thesis Context

| Item | Value |
|------|-------|
| Project name | **LISBOA** |
| Thesis Title | *LISBOA: Lisbon Itinerary System Based On AI* |
| Thesis Subtitle | *A Multi-Agent Approach for Personalized Tourism and Urban Mobility in Lisbon* |
| Author | André Filipe Gomes Silvestre, 20240502 |
| Supervisor | Prof. Dr. Bruno Jardim |
| Institution | NOVA IMS |
| Academic year | 2025/2026 |

## 📊 Repository Snapshot (as of 2026-04)

| Category | Current implementation |
|----------|------------------------|
| Supported Streamlit entrypoint | `app.py` |
| Runtime mode | Multi-agent by default |
| Specialized agents | Supervisor, Weather, Transport, Researcher, Planner, QA |
| Exported tools | 45 |
| Worker-agent tool assignment | Weather $4$, Transport $30$, Researcher $11$ |
| Knowledge base | ChromaDB with `BAAI/bge-m3` embeddings |
| Indexed collections | `lisbon_pdf`, `lisbon_places`, `lisbon_events` |
| Evaluation corpus | 72 benchmark queries across 6 domains |
| Evaluation artefacts | benchmark, ablation, coverage, and calibration outputs under `eval/results/` |
| Automation | daily scraping plus workflow-triggered vector sync |

## ✨ Core Capabilities

### 🌦️ Weather and Alerts

- current weather summaries
- 5-day forecasts
- Portugal-wide weather overview
- active meteorological warning retrieval

### 🚇 Mobility

- Metro de Lisboa status, wait times, frequencies, and nearest stations
- Carris Metropolitana alerts, routes, live locations, and departures
- Carris Urban buses and trams through GTFS and GTFS-RT
- CP schedules, routes, trip planning, and frequency support
- multimodal summaries and routing across providers

### 📚 Knowledge and Local Services

- semantic search over VisitLisboa places, events, and tourism knowledge
- on-demand Lisboa Aberta service discovery
- web fallback for history and culture queries

### 🧭 Planning and Synthesis

- constraint-aware itinerary synthesis
- integration of weather, transport, and local knowledge
- user-context support for interests, mobility, location, and available time

### 🧪 Evaluation and Research Support

- benchmark and ablation runners under `eval/`
- strict live coverage for the exported tool registry
- calibration support for human-vs-judge comparison
- reproducibility metadata and optional cost accounting

## 🤖 Why the System is Multi-Agent

*LISBOA* is no longer documented as a single monolithic ReAct agent. The supported runtime separates responsibilities into clearer layers:

- **Supervisor** for routing and direct handling of simple cases
- **Worker agents** for weather, transport, and research retrieval
- **QA validation** for completeness and factual safeguards
- **Planner synthesis** for itinerary-style responses

This reduces tool overload per worker, keeps domain prompts narrower, and makes final response assembly easier to control and evaluate.

## 🧱 Repository Highlights

| Path | Role |
|------|------|
| `app.py` | supported Streamlit UI entrypoint |
| `agent/` | orchestration, prompts, state, and shared agent utilities |
| `tools/` | exported tool registry plus vector-store internals |
| `data_collection/` | scrapers and source-acquisition scripts |
| `data/` | vector store and local transport support data |
| `eval/` | benchmarking, ablation, validators, and calibration assets |
| `tests/` | smoke checks, QA integration tests, and coverage validation |

## 📌 Documentation Boundary

> [!NOTE]
> This overview intentionally documents the supported application path around `app.py` and the current runtime architecture.
> Auxiliary thesis materials may exist in the repository, but they are not treated as the public operating path unless explicitly stated.
