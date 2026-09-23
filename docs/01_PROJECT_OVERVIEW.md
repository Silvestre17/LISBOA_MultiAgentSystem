# 📍 LISBOA Project Overview

LISBOA is the software artifact of a Master's thesis on grounded, context-aware assistance for tourism and urban mobility in Lisbon and the Lisbon Metropolitan Area (AML). It combines multi-agent orchestration, live and structured data integrations, municipal open data, and hybrid semantic and structured retrieval, served through the Streamlit entry point `app.py`.

## 👥 Who LISBOA Serves

| Audience | Typical Needs | Main Data Sources |
|---|---|---|
| **🧳 Tourists** | Itineraries, attractions, events, weather, and transport between landmarks | VisitLisboa, the Lisboa Card guide, IPMA, Metro de Lisboa, Carris Urban, CP, and multimodal routing |
| **🏠 Residents** | Daily mobility, nearby public services, local events, and municipal information | Lisboa Aberta, IPMA, Metro de Lisboa, Carris Urban, Carris Metropolitana, and CP |
| **🧪 Researchers and developers** | Reproducible architecture, grounded-tool design, evaluation, and ablation | Source code, technical documentation, evaluation corpus, validators, results, and notebooks |

## 🎓 Research Context

| Item | Value |
|---|---|
| Project | **LISBOA** (*Lisbon Itinerary System Based On AI*) |
| Thesis title | *LISBOA: A Multi-Agent Approach for Personalized Tourism and Urban Mobility in Lisbon* |
| Author | André Filipe Gomes Silvestre |
| Supervisors | Prof. Dr. Bruno Jardim and Prof. Dr. Miguel de Castro Neto |
| Degree | Master's in Data Science and Advanced Analytics, specialization in Data Science |
| Institution | NOVA Information Management School (NOVA IMS), Universidade NOVA de Lisboa |
| Academic year | 2025/2026 |
| Evaluated version | Commit [`9235dd3`](https://github.com/Silvestre17/LISBOA_MultiAgentSystem/tree/9235dd3) (May 15, 2026) |

The thesis research question is:

> How can an LLM-powered agent system effectively integrate diverse, real-time urban data sources (including transport status, weather, location, and schedule) to produce feasible, personalized, and context-aware tourist and mobility itineraries in Lisbon?

## 📊 Current System Snapshot

| Category | Implemented State |
|---|---|
| Runtime | `MultiAgentAssistant`, served through `app.py` |
| Agent roles | Supervisor, Weather, Transport, Researcher, Quality Assurance, and Planner |
| Specialist workers | Weather, Transport, and Researcher |
| Exported tools | **45**: Weather 4, Transport 30, and Researcher 11 |
| Knowledge base | ChromaDB with `BAAI/bge-m3` embeddings |
| Vector collections | `lisbon_pdf`, `lisbon_places`, and `lisbon_events` |
| Evaluation corpus | **72** scenarios across 6 domains: weather 13, transport 36, researcher 13, multi-agent 3, greeting 3, and out-of-scope 4 |
| Automation | 4 workflows: VisitLisboa refresh, vector sync, transport runtime assets, and Hugging Face deployment |

## ✨ Core Capabilities

| Domain | Implemented Coverage |
|---|---|
| 🌦️ **Weather** | Today's IPMA forecast summary, forecasts within the five-day provider horizon, a Portugal-wide overview, and active warnings |
| 🚇 **Mobility** | Status and routing for Metro de Lisboa, Carris Urban, Carris Metropolitana, CP suburban rail, and supported multimodal connections |
| 📚 **Tourism and services** | Hybrid VisitLisboa retrieval, Lisboa Card knowledge, on-demand Lisboa Aberta service discovery, and a constrained web fallback |
| 🧭 **Planning** | Itineraries built from gathered evidence and from the constraints stated in the request or its follow-ups, such as places, timing, mobility needs, weather, and transport preferences |
| 🧪 **Evaluation** | Isolated worker benchmark, zero-shot versus full-system ablation, dual-judge scoring, deterministic integrity checks, paired statistics, and a separate formative user study |

## 🤖 Why a Multi-Agent Approach

The runtime separates routing, domain retrieval, validation, and synthesis:

- `SupervisorAgent` interprets the request, answers direct cases, and selects the workers.
- The Weather, Transport, and Researcher workers have narrower prompts and tool sets.
- Deterministic checks preserve structured, source-backed outputs; generative QA runs only when required and can guide one targeted retry.
- `PlannerAgent` synthesizes itineraries when the evidence is sufficient; guarded structured fallbacks cover blocked or failed synthesis.

This division reduces the number of tools each worker must choose from and makes routing, evidence use, retries, and final responses easier to inspect and evaluate.

## 🧱 Repository Map

| Path | Role |
|---|---|
| `app.py` | Streamlit interface and entry point |
| `agent/` | Orchestration, agent roles, prompts, state, planning, and formatting |
| `tools/` | Exported tools plus location, release, and vector-store support |
| `data_collection/` | Source-acquisition scripts and static source documents |
| `data/` | Local or release-hydrated vector and transport runtime data |
| `eval/` | Evaluation corpus, benchmark, ablation, judges, statistics, and notebooks |
| `scripts/` | Smoke tests, provider checks, data publishing, and hosted startup |
| `docs/` | Technical documentation |

> [!NOTE]
> LISBOA is a Lisbon-focused proof of concept, not a general city benchmark. Availability and freshness depend on the implemented providers and their upstream services.
