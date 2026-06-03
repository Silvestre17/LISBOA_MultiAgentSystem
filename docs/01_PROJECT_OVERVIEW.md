# 📍 LISBOA Project Overview

LISBOA is a Master's thesis project for the Lisbon Metropolitan Area that combines multi-agent orchestration, provider-backed data integrations, municipal open data, and semantic retrieval. The documented public entrypoint is `app.py`.

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
| Author | André Filipe Gomes Silvestre |
| Supervisors | Prof. Dr. Bruno Jardim; Prof. Dr. Miguel de Castro Neto |
| Institution | NOVA IMS |
| Academic year | 2025/2026 |


| Category | Current implementation |
|----------|------------------------|
| Supported Streamlit entrypoint | `app.py` |
| Runtime mode | Multi-agent by default |
| Specialized agents | Supervisor, Weather, Transport, Researcher, Planner, QA |
| Exported tools | **45** |
| Worker-agent tool assignment | Weather **4**, Transport **30**, Researcher **11** |
| Knowledge base | ChromaDB with `BAAI/bge-m3` embeddings |
| Indexed collections | `lisbon_pdf`, `lisbon_places`, `lisbon_events` |
| Evaluation corpus | **72** ground-truth queries across 6 domains |
| Evaluation artefacts | benchmark, ablation, statistics, and figure outputs under `eval/results/` |
| Automation | daily scraping plus workflow-triggered vector sync |

## ✨ Core Capabilities

| Domain | What it covers |
|--------|----------------|
| 🌦️ **Weather & alerts** | current summary, 5-day forecast, Portugal-wide overview, active IPMA warnings |
| 🚇 **Mobility** | Metro de Lisboa status / wait times / nearest station; Carris Metropolitana alerts, routes, live positions, departures; Carris Urban GTFS + GTFS-RT; CP schedules, routes, trip planning; multimodal summaries and routing |
| 📚 **Knowledge & local services** | semantic search over VisitLisboa places & events; on-demand Lisboa Aberta service discovery; web fallback for history & culture |
| 🧭 **Planning & synthesis** | constraint-aware itineraries integrating weather, transport, and user context (interests, mobility, location, available time) |
| 🧪 **Evaluation & research** | benchmark and ablation runners under `eval/`; deterministic dataset and validator integrity checks; statistical analysis; reproducibility metadata and optional cost accounting |

## 🤖 Why the System is Multi-Agent

The supported runtime separates responsibilities into clearer layers — **Supervisor** (routing and direct handling), **worker agents** (weather, transport, research), **QA validation** (completeness and factual safeguards), and **Planner synthesis** (itinerary responses). This reduces tool overload per worker, keeps domain prompts narrower, and makes final response assembly easier to control and evaluate.

## 🧱 Repository Highlights

| Path | Role |
|------|------|
| `app.py` | Supported Streamlit UI entrypoint |
| `agent/` | Orchestration, prompts, state, and shared agent utilities |
| `tools/` | Exported tool registry plus vector-store internals |
| `data_collection/` | Scrapers and source-acquisition scripts |
| `data/` | Vector store and local transport support data |
| `eval/` | Benchmarking, ablation, validators, and statistical analysis assets |
| `scripts/` | Operational helpers (syntax check, prompt smoke runner, transport verification) |
| `docs/` | This documentation set |

## 📌 Documentation Boundary

> [!NOTE]
> This overview intentionally documents the supported application path around `app.py` and the current runtime architecture. Auxiliary thesis materials may exist in the repository, but they are not treated as the public operating path unless explicitly stated.
