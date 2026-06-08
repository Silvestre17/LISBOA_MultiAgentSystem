<p align="center">
  <a href="https://github.com/Silvestre17/LISBOA_MultiAgentSystem">
    <img src="./img/BannerLSIBOA_21-9.png" alt="LISBOA Project Banner" style="width: 100%; height: auto;">
  </a>
</p>

# 🗺️ LISBOA: Lisbon Itinerary System Based On AI 🤖

<p align="center">
  <a href="https://github.com/Silvestre17/LISBOA_MultiAgentSystem"><img src="https://img.shields.io/badge/Project_Repo-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repo"></a>
  <a href="https://silvestre17.github.io/LISBOA_MultiAgentSystem/"><img src="https://img.shields.io/badge/Streamlit_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit App"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/LangGraph-Multi--Agent-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph Multi-Agent">
  <img src="https://img.shields.io/badge/Exported_Tools-45-0A7E07?style=for-the-badge" alt="45 exported tools">
</p>

<p align="center">
  <strong>Multi-agent assistance for Lisbon tourism and urban mobility, grounded in RAG, provider-backed data integrations, municipal open data, and a research evaluation pipeline.</strong>
</p>

<p align="center">
  <a href="https://www.visitlisboa.com/"><img src="https://img.shields.io/badge/VisitLisboa-FED30E?style=for-the-badge" alt="VisitLisboa"></a>
  <a href="https://dados.gov.pt/"><img src="https://img.shields.io/badge/Lisboa_Aberta-EF7E22?style=for-the-badge" alt="Lisboa Aberta"></a>
  <a href="https://api.ipma.pt/"><img src="https://img.shields.io/badge/IPMA-257ABA?style=for-the-badge" alt="IPMA"></a>
  <a href="https://www.metrolisboa.pt/"><img src="https://img.shields.io/badge/Metro_de_Lisboa-EF5A34?style=for-the-badge" alt="Metro de Lisboa"></a>
  <a href="https://www.carrismetropolitana.pt/"><img src="https://img.shields.io/badge/Carris_Metropolitana-FFDD00?style=for-the-badge" alt="Carris Metropolitana"></a>
  <a href="https://www.carris.pt/"><img src="https://img.shields.io/badge/Carris-00468F?style=for-the-badge" alt="Carris"></a>
  <a href="https://www.cp.pt/"><img src="https://img.shields.io/badge/CP-388344?style=for-the-badge" alt="CP"></a>
</p>


<a id="overview"></a>
## 📍 Overview

LISBOA is a Master's thesis project at NOVA IMS that implements a multi-agent system for personalized tourist planning and urban mobility support in the Lisbon Metropolitan Area. It combines Retrieval-Augmented Generation (RAG), weather and transport integrations, municipal open data, and a Streamlit interface to support grounded, context-aware answers.

> [!IMPORTANT]
> The supported user-facing entrypoint is `app.py`. The runtime is the multi-agent system implemented by `MultiAgentAssistant` in [`agent/graph.py`](./agent/graph.py).

<p align="center">
  <img src="./img/LISBOA_Framework.png" alt="LISBOA framework figure" width="720">
</p>

<a id="quick-links"></a>
## 🔗 Quick Links

- [👥 Who the System Serves](#who-the-system-serves)
- [📊 Current System Snapshot](#current-system-snapshot)
- [🏗️ System Architecture](#system-architecture)
- [🖼️ Framework Figure](./img/LISBOA_Framework.png)
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
- **Supervisors:** Prof. Dr. Bruno Jardim; Prof. Dr. Miguel de Castro Neto
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
| Evaluation artefacts | benchmark, ablation, statistics, and figure outputs under `eval/results/` |
| Automation workflows | `data_pipeline.yml` and `sync_vector_db.yml` |

## ✨ Core Capabilities

- 🌦️ **Weather** (IPMA): warnings, 5-day forecast, current summary, Portugal-wide overview
- 🚇 **Mobility**: Metro de Lisboa, Carris Metropolitana, Carris Urban, CP, and multimodal routing
- 📚 **Local knowledge**: VisitLisboa events/places, Lisboa Aberta open data, indexed Lisbon guide PDF, web fallback
- 🧭 **Itinerary synthesis** with weather, transport, and preference grounding
- ✅ **QA validation** before final answers; prompt smoke validation for user-facing changes

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

1. `SupervisorAgent.route()` decides whether to answer directly or invoke workers.
2. Workers run **in parallel** when the query spans multiple domains.
3. `QualityAssuranceAgent.validate()` enforces completeness, factual consistency, and language alignment.
4. For planning queries, `PlannerAgent.synthesize()` writes the final itinerary; otherwise the supervisor or combined worker output is returned directly.

> The planner is **not** the universal final responder — it only synthesizes itineraries.

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
- **Evaluation:** deterministic dataset/validator suites under `eval/tests/`, plus benchmark and ablation runners
- **Automation:** GitHub Actions for scraping and vector synchronization

<a id="data-sources-and-tool-inventory"></a>
## 🌐 Data Sources and Tool Inventory

**45 exported LangChain tools** (`tools/__init__.py`), grouped by domain:

| Domain | Tools | Source |
|--------|------:|--------|
| Weather | **4** | IPMA |
| Metro de Lisboa | **6** | Official API + public fallback |
| Carris Metropolitana | **8** | REST API |
| Carris Urban | **8** | GTFS + GTFS-RT |
| CP / Comboios.live | **6** | Live API + local GTFS |
| Multimodal routing | **2** | Composed cross-provider |
| VisitLisboa | **5** | Scraped JSON + ChromaDB |
| Lisboa Aberta | **5** | GeoJSON open data |
| Web fallback | **1** | Tavily search |

The Lisbon guide PDF is served through internal vector search (not a separate exported tool). Tool counts can change; verify `tools/__init__.py` before making exact thesis or documentation claims.

→ Detail: [`docs/03_TOOLS_REFERENCE.md`](./docs/03_TOOLS_REFERENCE.md) · [`docs/04_DATA_SOURCES_AND_SCHEMAS.md`](./docs/04_DATA_SOURCES_AND_SCHEMAS.md)

<a id="evaluation-and-research-workflow"></a>
## 🧪 Evaluation and Research Workflow

Research-grade stack under `eval/` combining LLM-as-a-Judge, deterministic metrics, prompt smoke validation, and statistical analysis.

| Layer | Entrypoint | Output |
|------|-----------|--------|
| Fast deterministic checks | `eval/tests/` | test output only |
| Benchmark (isolated workers) | `eval/run_benchmark.py` | `eval/results/benchmark/` |
| Ablation (zero-shot vs LISBOA) | `eval/run_ablation.py` | `eval/results/ablation/` |
| Prompt smoke validation | `scripts/run_prompts.py` | terminal output / chosen artefacts |

**Ground truth**: 72 entries across 6 domains — weather (13), transport (36), researcher (13), multi-agent (3), greeting (3), out-of-scope (4).

**Measured**: factual accuracy, tool usage, completeness, relevance, response quality (LLM-as-a-Judge); tool *P/R/F1*, response heuristics, deterministic Metro route validation; reproducibility metadata, token usage, and optional cost accounting.

→ Full schema and methodology: [`eval/README.md`](./eval/README.md) · Notebook: [`eval/benchmark_ablation_analysis.ipynb`](./eval/benchmark_ablation_analysis.ipynb)

## 🧱 Repository Structure

```text
LISBOA_MultiAgentSystem/
├── agent/                          # Multi-agent orchestration, prompts, utilities
├── tools/                          # 45 exported LangChain tools + vector store internals
├── data_collection/                # Scrapers and data acquisition scripts
├── data/                           # Persistent vector DB and local transport data
├── docs/                           # Repository documentation
├── eval/                           # Benchmarking, ablation, judge, validators, statistics
├── eval/tests/                     # Lean deterministic checks and dataset validators
├── .github/workflows/              # Scraping and vector sync automation
├── app.py                          # Supported Streamlit entrypoint
├── config.py                       # Runtime configuration and provider selection
├── pyproject.toml                  # Package metadata and local package discovery
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
| [`eval/README.md`](./eval/README.md) | Evaluation pipeline, benchmark logic, live coverage, and artefact structure |
| [`eval/benchmark_ablation_analysis.ipynb`](./eval/benchmark_ablation_analysis.ipynb) | Analysis notebook for benchmark and ablation outputs |

### Suggested Reading Paths

- **New to the repository:** [`docs/00_INDEX.md`](./docs/00_INDEX.md) -> [`docs/01_PROJECT_OVERVIEW.md`](./docs/01_PROJECT_OVERVIEW.md) -> [`docs/02_SYSTEM_ARCHITECTURE.md`](./docs/02_SYSTEM_ARCHITECTURE.md)
- **Need the exact capabilities:** [`docs/03_TOOLS_REFERENCE.md`](./docs/03_TOOLS_REFERENCE.md) -> [`docs/04_DATA_SOURCES_AND_SCHEMAS.md`](./docs/04_DATA_SOURCES_AND_SCHEMAS.md)
- **Want to run it locally:** [`docs/05_DEPLOYMENT_AND_OPERATIONS.md`](./docs/05_DEPLOYMENT_AND_OPERATIONS.md) -> [`.env.example`](./.env.example) -> [Getting started](#getting-started)
- **Working on evaluation:** [`eval/README.md`](./eval/README.md) -> [`eval/benchmark_ablation_analysis.ipynb`](./eval/benchmark_ablation_analysis.ipynb)

<a id="getting-started"></a>
## 🚀 Getting Started

**Prerequisites**: Python 3.10+, Git, and one configured LLM provider (***Azure OpenAI***, ***OpenAI***, or ***LM Studio***). ***Metro*** credentials and a ***Tavily API*** key are optional.

```bash
# 1. Clone
git clone https://github.com/Silvestre17/LISBOA_MultiAgentSystem.git
cd LISBOA_MultiAgentSystem

# 2. Install the supported runtime
pip install -r requirements.txt
# ...or full local env (scraping, eval, notebooks, CUDA-enabled PyTorch):
# conda env create -f environment_local_gpu.yml && conda activate lisboa_thesis2026

# 3. Configure secrets
cp .env.example .env     # Windows PowerShell: Copy-Item .env.example .env

# 4. Build vector store and launch
python tools/vector_store.py
streamlit run app.py
```

Full provider, tracing, and TLS notes: [`docs/05_DEPLOYMENT_AND_OPERATIONS.md`](./docs/05_DEPLOYMENT_AND_OPERATIONS.md).

## ✅ Testing and Evaluation

```bash
# Fast deterministic checks
python scripts/syntax_check.py
python -m pytest eval/tests/ -q

# Single-prompt smoke test
python scripts/run_prompts.py --suite smoke
python scripts/run_prompts.py --prompt "How do I get from Baixa-Chiado to Aeroporto?" --language en --quiet

# Benchmark / ablation (module form required)
python -m eval.run_benchmark --mode run_test
python -m eval.run_ablation  --mode run_test
```

> [!IMPORTANT]
> Benchmark and ablation runners require module form (`python -m eval.run_benchmark`). Direct script invocation breaks `agent` import resolution.

Artefacts land under `eval/results/{benchmark,ablation,statistics,figures}/`. See [`eval/README.md`](./eval/README.md) for the current validation policy.

## Known Limitations

- LISBOA is a research prototype, not a production travel, booking, ticketing, reservation, or transaction service.
- Live or current data is available only where the implemented provider integrations support it. VisitLisboa content, local vector stores, and transport runtime assets may be cached, scraped, scheduled, or release-based.
- Mobility coverage is limited to implemented Lisbon/AML operators and tools: Metro de Lisboa, Carris Urban, Carris Metropolitana, CP suburban rail, and the repository's supported multimodal routing logic.
- The planner synthesizes evidence gathered by the worker agents. It does not independently verify facts beyond the repository's QA and formatting guardrails.
- Public evaluation artifacts cover the automated benchmark, ablation, statistics, figures, and deterministic checks included under `eval/`. User-study material should be treated as separate unless explicitly published.

## ⚙️ Automation

Two GitHub Actions workflows keep the knowledge base fresh:

1. [`data_pipeline.yml`](./.github/workflows/data_pipeline.yml) scrapes VisitLisboa content **daily at 04:00 Europe/Lisbon time**. Places are refreshed weekly on **Mondays** during scheduled runs. Manual runs can choose `events`, `places`, or `both` without changing the automatic behaviour.
2. [`sync_vector_db.yml`](./.github/workflows/sync_vector_db.yml) runs after the scraping workflow completes successfully and performs incremental vector synchronization.

> [!NOTE]
> Both workflows can also be triggered manually from the GitHub Actions tab.

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](./LICENSE) for details.

### Citation

If you use this repository before a final thesis, paper, or DOI-based citation is available, please cite it as software using APA 7th edition:

> Silvestre, A. (2026). *LISBOA: Lisbon itinerary system based on AI: A multi-agent approach for personalized tourism and urban mobility in Lisbon* [Computer software]. GitHub. https://github.com/Silvestre17/LISBOA_MultiAgentSystem

A BibTeX entry is also provided for convenience:

```bibtex
@software{silvestre2026lisboa,
  author = {Silvestre, André},
  title = {LISBOA: Lisbon Itinerary System Based On AI: A Multi-Agent Approach for Personalized Tourism and Urban Mobility in Lisbon},
  year = {2026},
  type = {Computer software},
  publisher = {GitHub},
  url = {https://github.com/Silvestre17/LISBOA_MultiAgentSystem}
}
```

### Responsible Use and Privacy

LISBOA is a research prototype developed for academic evaluation and demonstration purposes. Users should verify operational decisions, including departures, disruptions, opening hours, prices, accessibility conditions, and ticket information, with the official providers before acting.

Do not enter sensitive personal data, credentials, private identifiers, or confidential information in prompts, logs, notebooks, or evaluation artifacts.

---

<p align="center">
  <i>Developed as part of the Master's Thesis in Data Science and Advanced Analytics at NOVA IMS (2025-2026)</i>
</p>

<p align="center">
  <a href="https://www.novaims.unl.pt/"><img src="https://img.shields.io/badge/NOVA_IMS-0ee071?style=for-the-badge&logo=university&logoColor=white" alt="NOVA IMS"></a>
</p>

