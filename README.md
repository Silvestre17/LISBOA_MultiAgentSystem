<p align="center">
  <a href="https://github.com/Silvestre17/Thesis2025-26_AFGS">
    <img src="./img/BannerLSIBOA_21-9.png" alt="LISBOA Project Banner" style="width: 100%; height: 300px; object-fit: cover; object-position: center bottom;">
  </a>
</p>

# 🗺️ LISBOA: Lisbon Itinerary System Based On AI 🤖

<p align="center">
  <a href="https://github.com/Silvestre17/Thesis2025-26_AFGS"><img src="https://img.shields.io/badge/Project_Repo-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repo"></a>
  <a href="#"><img src="https://img.shields.io/badge/Streamlit_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit App"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/LangGraph-Multi--Agent-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph Multi-Agent">
  <img src="https://img.shields.io/badge/Exported_Tools-45-0A7E07?style=for-the-badge" alt="45 exported tools">
</p>

<p align="center">
  <strong>Multi-agent assistance for Lisbon tourism and urban mobility, grounded in RAG, live APIs, municipal open data, and a research-grade evaluation pipeline.</strong>
</p>

<p align="center">
  <a href="https://www.visitlisboa.com/"><img src="https://img.shields.io/badge/VisitLisboa-FF6B35?style=for-the-badge" alt="VisitLisboa"></a>
  <a href="https://api.ipma.pt/"><img src="https://img.shields.io/badge/IPMA-0052CC?style=for-the-badge" alt="IPMA"></a>
  <a href="https://www.metrolisboa.pt/"><img src="https://img.shields.io/badge/Metro_de_Lisboa-E60000?style=for-the-badge" alt="Metro de Lisboa"></a>
  <a href="https://www.carrismetropolitana.pt/"><img src="https://img.shields.io/badge/Carris_Metropolitana-00A859?style=for-the-badge" alt="Carris Metropolitana"></a>
  <a href="https://www.cp.pt/"><img src="https://img.shields.io/badge/CP-003DA5?style=for-the-badge" alt="CP"></a>
</p>

<p align="center">
  <a href="./docs/00_INDEX.md"><img src="https://img.shields.io/badge/Docs_Index-0A66C2?style=for-the-badge" alt="Docs Index"></a>
  <a href="./docs/02_SYSTEM_ARCHITECTURE.md"><img src="https://img.shields.io/badge/System_Architecture-6A1B9A?style=for-the-badge" alt="System Architecture"></a>
  <a href="./docs/03_TOOLS_REFERENCE.md"><img src="https://img.shields.io/badge/Tools_Reference-8E24AA?style=for-the-badge" alt="Tools Reference"></a>
  <a href="./eval/README.md"><img src="https://img.shields.io/badge/Evaluation_Pipeline-C77800?style=for-the-badge" alt="Evaluation Pipeline"></a>
</p>

<a id="overview"></a>
## 📍 Overview

LISBOA is a Master's thesis project at NOVA IMS that implements an intelligent multi-agent system for personalized tourist planning and urban mobility support in the Lisbon Metropolitan Area. It combines Retrieval-Augmented Generation (RAG), real-time transport and weather APIs, municipal open data, and a Streamlit interface to support grounded, context-aware answers.

The supported user-facing entrypoint is `app.py`. The current runtime is multi-agent by default (`Config.USE_MULTI_AGENT = True`).

> Start here for the guided repo tour: [`docs/00_INDEX.md`](./docs/00_INDEX.md)

<a id="quick-links"></a>
## 🔗 Quick Links

- [👥 Who the System Serves](#who-the-system-serves)
- [📊 Current System Snapshot](#current-system-snapshot)
- [🏗️ System Architecture](#system-architecture)
- [🌐 Data Sources and Tool Inventory](#data-sources-and-tool-inventory)
- [🧪 Evaluation and Research Workflow](#evaluation-and-research-workflow)
- [📚 Documentation Hub](#documentation-hub)
- [🚀 Getting Started](#getting-started)
- [📘 Docs Index](./docs/00_INDEX.md)
- [🧭 Architecture Doc](./docs/02_SYSTEM_ARCHITECTURE.md)
- [🛠️ Tools Reference](./docs/03_TOOLS_REFERENCE.md)
- [⚙️ Operations Guide](./docs/05_DEPLOYMENT_AND_OPERATIONS.md)
- [📊 Evaluation README](./eval/README.md)

<a id="who-the-system-serves"></a>
## 👥 Who the System Serves

| Audience | Typical questions | Main data layers |
|----------|-------------------|------------------|
| Tourists | itineraries, museums, events, weather, transport between landmarks | VisitLisboa, IPMA, Metro, Carris, CP, multimodal routing |
| Residents | daily transport, nearby services, local events, open urban data | Lisboa Aberta, Metro, Carris Metropolitana, Carris Urban, CP, IPMA |

<a id="project-context"></a>
## 🎓 Project Context

- **Thesis title:** *LISBOA: Lisbon Itinerary System Based On AI*
- **Subtitle:** *A Multi-Agent Approach for Personalized Tourism and Urban Mobility in Lisbon*
- **Author:** André Filipe Gomes Silvestre, 20240502
- **Supervisor:** Prof. Dr. Bruno Jardim
- **Institution:** NOVA IMS, Master's in Data Science and Advanced Analytics
- **Academic year:** 2025/2026

<a id="current-system-snapshot"></a>
## 📊 Current System Snapshot

| Item | Current state |
|------|---------------|
| Supported UI entrypoint | `app.py` |
| Runtime mode | Multi-agent |
| Specialized agents | 6 total: Supervisor, Weather, Transport, Researcher, Planner, QA |
| Exported LangChain tools | 45 |
| Transport tool set | 30 tools |
| Researcher tool set | 11 tools |
| Vector collections | 3: `lisbon_pdf`, `lisbon_places`, `lisbon_events` |
| Evaluation ground truth | 72 benchmark queries across 6 domains |
| Evaluation artefacts | benchmark, ablation, coverage, calibration outputs under `eval/results/` |
| Automation workflows | `data_pipeline.yml` and `sync_vector_db.yml` |

## ✨ Core Capabilities

LISBOA currently supports:

- 🌦️ Real-Time Weather queries through IPMA
- 🚇 Real-Time Transport queries across Metro de Lisboa, Carris Metropolitana, Carris Urban, CP, and multimodal routing
- 📚 Real-Time Tourism and local knowledge retrieval via *VisitLisboa*, *Lisboa Aberta*, semantic vector search, and web fallback search
- 🧭 Itinerary synthesis that combines user preferences, timing, mobility constraints, and live operational context
- ✅ Quality Assurance validation before final multi-step responses are returned
- 🧪 Evaluation workflows for benchmark, ablation, strict live tool coverage, and calibration

<a id="system-architecture"></a>
## 🏗️ System Architecture

The default application flow is orchestrated by `MultiAgentAssistant` in `agent/graph.py`.

> Deep dive: [`docs/02_SYSTEM_ARCHITECTURE.md`](./docs/02_SYSTEM_ARCHITECTURE.md)

```text
User query
  -> SupervisorAgent
  -> Specialized agents in parallel when needed
  -> QualityAssuranceAgent validation
  -> PlannerAgent synthesis for planning requests, or direct combined response
```

### Runtime Roles

| Agent | Role | Tools | Notes |
|------|------|------:|------|
| `SupervisorAgent` | intent classification, routing, direct responses for simple cases | 0 | handles greetings and out-of-scope requests directly |
| `WeatherAgent` | IPMA weather retrieval | 4 | weather specialist with tool-calling flow |
| `TransportAgent` | transport retrieval | 30 | covers Metro, Carris Metropolitana, Carris Urban, CP, and multimodal routing |
| `ResearcherAgent` | tourism knowledge, open data, web fallback | 11 | combines VisitLisboa, Lisboa Aberta, PDF knowledge, and web search |
| `QualityAssuranceAgent` | completeness and factual validation | 0 | validates worker outputs and can request retry paths |
| `PlannerAgent` | final synthesis for itinerary requests | 0 | only produces the final itinerary when planning is required |

### Response Flow

1. `SupervisorAgent.route()` decides whether the request should be handled directly or forwarded to worker agents.
2. Worker agents run in parallel when the query spans multiple domains.
3. `QualityAssuranceAgent.validate()` checks completeness, factual consistency, and user-context alignment.
4. If the route includes planning, `PlannerAgent.synthesize()` creates the final itinerary.
5. Simple or single-domain requests may bypass the planner and return directly from the supervisor or from combined worker outputs.

This means the planner is **not** the universal final responder. It is the final synthesizer **for planning queries**.

## 🧰 Technology Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/LangChain-Orchestration-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/LangGraph-Agent_Graph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph">
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B6B?style=for-the-badge" alt="ChromaDB">
  <img src="https://img.shields.io/badge/BAAI%2Fbge--m3-Multilingual_Embeddings-FFC700?style=for-the-badge&logo=huggingface&logoColor=black" alt="BAAI bge-m3">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
</p>

- **LLM providers supported:** Azure OpenAI, OpenAI, LM Studio
- **Embedding model:** `BAAI/bge-m3`
- **Packaging:** `pyproject.toml` supports editable installs
- **Testing:** `pytest` suites under `tests/` and `eval/tests/`
- **Automation:** GitHub Actions for scraping and vector synchronization

<a id="data-sources-and-tool-inventory"></a>
## 🌐 Data Sources and Tool Inventory

### Live APIs

| Source | Tools | Coverage |
|--------|------:|----------|
| IPMA | 4 | warnings, 5-day forecast, current summary, Portugal-wide overview |
| Metro de Lisboa | 6 | line status, station wait times, line wait times, nearest station, frequencies, station list |
| Carris Metropolitana | 8 | alerts, stops, lines, route discovery, live positions, departures |
| Carris Urban | 8 | GTFS and GTFS-RT routes, stops, arrivals, realtime vehicles, ETA, frequency |
| CP / Comboios.live | 6 | train status, station search, schedules, routes, trip planning, frequency |
| Multi-modal transport | 2 | cross-provider status and routing |

### Indexed and On-Demand Knowledge

| Source | Tools | Coverage |
|--------|------:|----------|
| VisitLisboa | 5 | places, attractions, events, categories, semantic tourism search |
| Lisboa Aberta | 5 | nearby services, dataset metadata, category browsing, place search |
| Lisbon guide PDF | internal vector search | semantic retrieval from the indexed guide |
| Web knowledge fallback | 1 | Lisbon history and culture search |

The authoritative export registry lives in `tools/__init__.py`, and the strict live coverage manifest in `tests/fixtures/tool_coverage_manifest.json` ensures every exported tool is referenced at least once in evaluation assets.

More detail:

- [`docs/03_TOOLS_REFERENCE.md`](./docs/03_TOOLS_REFERENCE.md)
- [`docs/04_DATA_SOURCES_AND_SCHEMAS.md`](./docs/04_DATA_SOURCES_AND_SCHEMAS.md)

<a id="evaluation-and-research-workflow"></a>
## 🧪 Evaluation and Research Workflow

The evaluation stack lives in `eval/` and combines LLM judgment, deterministic metrics, strict live coverage, and calibration support.

> Full workflow and output schema details: [`eval/README.md`](./eval/README.md)

The companion evaluation guide also documents optional coverage and calibration artefacts created on demand, the latest CSV exports from the analysis notebook, and figure outputs under `eval/results/figures/`.

### Evaluation Layers

| Layer | Purpose | Main entrypoints | Output location |
|------|---------|------------------|-----------------|
| Fast deterministic checks | validate datasets, utilities, validators, judge helpers, and the coverage manifest | `eval/tests/`, especially `eval/tests/test_dataset_integrity.py` | test output only |
| Benchmark | evaluate isolated worker agents | `eval/run_benchmark.py` | `eval/results/benchmark/` |
| Ablation | compare zero-shot vs LISBOA pipeline | `eval/run_ablation.py` | `eval/results/ablation/` |
| Strict live coverage | verify real use of the exported tool registry | `tests/test_tool_prompt_coverage.py` | `eval/results/coverage/` |
| Calibration | compare human scores vs judge scores | `eval/human_calibration/run_calibration.py` | `eval/results/calibration/` |

### Ground Truth Coverage

72 evaluation entries across 6 domains:

| Domain | Count |
|--------|------:|
| `weather` | 13 |
| `transport` | 36 |
| `researcher` | 13 |
| `multi_agent` | 3 |
| `greeting` | 3 |
| `out_of_scope` | 4 |

### What the Pipeline Measures

- 📏 LLM-as-a-Judge scoring for factual accuracy, tool usage, completeness, relevance, and response quality
- 🧮 Deterministic tool *Precision*, *Recall*, and *F1*
- 🛡️ Response heuristics such as tool leaks, language compliance, response length, hallucinated feature detection, and emoji density
- 🚇 deterministic Metro route validation for transport answers
- 💰 reproducibility metadata, token usage, and optional cost accounting
- 📓 benchmark and ablation analysis via [`eval/benchmark_ablation_analysis.ipynb`](./eval/benchmark_ablation_analysis.ipynb)

## 🧱 Repository Structure

```text
Thesis2025-26_AFGS/
├── agent/                          # Multi-agent orchestration, prompts, utilities
├── tools/                          # 45 exported LangChain tools + vector store internals
├── data_collection/                # Scrapers and data acquisition scripts
├── data/                           # Persistent vector DB and local transport data
├── docs/                           # Repository documentation
├── eval/                           # Benchmarking, ablation, judge, validators, calibration
├── tests/                          # Runtime tests, smoke suites, live coverage manifests
├── .github/workflows/              # Scraping and vector sync automation
├── app.py                          # Supported Streamlit entrypoint
├── config.py                       # Runtime configuration and provider selection
├── pyproject.toml                  # Package metadata and optional extras
└── README.md                       # Project overview
```

Need a guided reading order? Open [`docs/00_INDEX.md`](./docs/00_INDEX.md).

<a id="documentation-hub"></a>
## 📚 Documentation Hub

| Document | Purpose |
|----------|---------|
| [`docs/00_INDEX.md`](./docs/00_INDEX.md) | Start here, navigation hub for the full repository documentation |
| [`docs/01_PROJECT_OVERVIEW.md`](./docs/01_PROJECT_OVERVIEW.md) | Scope, audiences, current snapshot, and project framing |
| [`docs/02_SYSTEM_ARCHITECTURE.md`](./docs/02_SYSTEM_ARCHITECTURE.md) | Agent topology, orchestration, and runtime design |
| [`docs/03_TOOLS_REFERENCE.md`](./docs/03_TOOLS_REFERENCE.md) | Exact tool inventory and agent-to-tool mapping |
| [`docs/04_DATA_SOURCES_AND_SCHEMAS.md`](./docs/04_DATA_SOURCES_AND_SCHEMAS.md) | Data sources, refresh cadences, schemas, and vector collections |
| [`docs/05_DEPLOYMENT_AND_OPERATIONS.md`](./docs/05_DEPLOYMENT_AND_OPERATIONS.md) | Environment setup, automation, troubleshooting, and operations |
| [`docs/06_FUTURE_ENHANCEMENTS.md`](./docs/06_FUTURE_ENHANCEMENTS.md) | Roadmap and next-step ideas |
| [`eval/README.md`](./eval/README.md) | Evaluation pipeline, benchmark logic, live coverage, and artefact structure |
| [`eval/benchmark_ablation_analysis.ipynb`](./eval/benchmark_ablation_analysis.ipynb) | Analysis notebook for benchmark and ablation outputs |

### Suggested Reading Paths

- **New to the repository:** [`docs/00_INDEX.md`](./docs/00_INDEX.md) -> [`docs/01_PROJECT_OVERVIEW.md`](./docs/01_PROJECT_OVERVIEW.md) -> [`docs/02_SYSTEM_ARCHITECTURE.md`](./docs/02_SYSTEM_ARCHITECTURE.md)
- **Need the exact capabilities:** [`docs/03_TOOLS_REFERENCE.md`](./docs/03_TOOLS_REFERENCE.md) -> [`docs/04_DATA_SOURCES_AND_SCHEMAS.md`](./docs/04_DATA_SOURCES_AND_SCHEMAS.md)
- **Want to run it locally:** [`docs/05_DEPLOYMENT_AND_OPERATIONS.md`](./docs/05_DEPLOYMENT_AND_OPERATIONS.md) -> [`.env.example`](./.env.example) -> [Getting started](#getting-started)
- **Working on evaluation:** [`eval/README.md`](./eval/README.md) -> [`eval/benchmark_ablation_analysis.ipynb`](./eval/benchmark_ablation_analysis.ipynb)

<a id="getting-started"></a>
## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Git**
- **One configured LLM provider**, either Azure OpenAI, OpenAI, or LM Studio
- **Optional Metro credentials** for the official Metro de Lisboa realtime endpoints

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Silvestre17/Thesis2025-26_AFGS.git
   cd Thesis2025-26_AFGS
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional editable install for tests and evaluation**
   ```bash
   pip install -e ".[eval,dev]"
   ```

### Environment Setup

Copy [`.env.example`](./.env.example) to `.env` and fill in only the providers and services you plan to use.

Common entries:

- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME`
- or `OPENAI_API_KEY`, `OPENAI_MODEL_NAME`
- `METRO_CONSUMER_KEY`, `METRO_CONSUMER_SECRET` for official Metro realtime data
- `TAVILY_API_KEY` for web search
- `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT` for optional LangSmith tracing
- optionally `LANGSMITH_WORKSPACE_ID` if the LangSmith API key is linked to multiple workspaces

LM Studio can also be used locally with no API key, as documented in [`.env.example`](./.env.example) and [`config.py`](./config.py).

Notes for tracing:

- the code still accepts legacy `LANGCHAIN_*` tracing aliases for backward compatibility, but `LANGSMITH_*` is now the recommended setup
- each real user message should produce exactly one top-level LangSmith trace, with supervisor, worker agents, and tool spans nested inside it
- the `Save & Connect` or provider health-check flows use raw HTTP requests on purpose and do **not** create LangSmith traces, so they do not consume the free-tier tracing quota

### First Run

1. **Build or refresh the vector store**
   ```bash
   python tools/vector_store.py
   ```

2. **Run the application**
   ```bash
   streamlit run app.py
   ```

More operational detail is available in [`docs/05_DEPLOYMENT_AND_OPERATIONS.md`](./docs/05_DEPLOYMENT_AND_OPERATIONS.md).

## ✅ Testing and Evaluation

### Fast Validation

```bash
python scripts/syntax_check.py
python -m pytest eval/tests/test_dataset_integrity.py eval/tests/test_benchmark_utils.py eval/tests/test_cost_accounting.py eval/tests/test_llm_judge.py eval/tests/test_validators.py -v
python -m pytest tests/test_qa_agent.py tests/test_audit_fixes.py tests/test_response_guardrails.py tests/test_transport_parity_and_rendering.py tests/test_langsmith_tracing.py tests/test_metro_api_fallback_messaging.py -q
python scripts/run_prompts.py --suite smoke
```

### Main Evaluation Commands

```bash
python eval/run_benchmark.py --mode run_test
python eval/run_benchmark.py --mode full
python eval/run_benchmark.py --limit 5
python eval/run_ablation.py --mode run_test
python eval/run_ablation.py --mode full
python -m pytest tests/test_tool_prompt_coverage.py --run-live -m "live and coverage" -v
```

### Artefact Locations

- benchmark outputs: `eval/results/benchmark/`
- ablation outputs: `eval/results/ablation/`
- strict live coverage outputs: `eval/results/coverage/` (created when the live suite runs)
- calibration outputs: `eval/results/calibration/` (created when calibration runs)
- notebook figure exports: `eval/results/figures/`

See [`eval/README.md`](./eval/README.md) for the complete evaluation workflow, output schemas, and environment prerequisites.

## ⚙️ Automation

Two GitHub Actions workflows keep the knowledge base fresh:

1. [`data_pipeline.yml`](./.github/workflows/data_pipeline.yml) scrapes VisitLisboa content daily at **04:00 UTC**. Places are refreshed weekly on Mondays unless manually triggered.
2. [`sync_vector_db.yml`](./.github/workflows/sync_vector_db.yml) runs after the scraping workflow completes successfully and performs incremental vector synchronization.

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](./LICENSE) for details.

---

<p align="center">
  <i>Developed as part of the Master's Thesis in Data Science and Advanced Analytics at NOVA IMS (2025-2026)</i>
</p>

<p align="center">
  <a href="https://www.novaims.unl.pt/"><img src="https://img.shields.io/badge/NOVA_IMS-0ee071?style=for-the-badge&logo=university&logoColor=white" alt="NOVA IMS"></a>
</p>

