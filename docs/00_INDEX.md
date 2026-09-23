# 📚 LISBOA Documentation Hub

This documentation describes the current `main` branch. Its statements are anchored in the code: `app.py`, `agent/graph.py`, `tools/__init__.py`, `config.py`, the GitHub Actions workflows, and the evaluation code.

> [!IMPORTANT]
> The supported entry point is `app.py`, launched with `streamlit run app.py`. Setup, configuration, and validation are covered in [Deployment and Operations](./05_DEPLOYMENT_AND_OPERATIONS.md).

## 🔎 At a Glance

| Item | Current Value |
|---|---|
| Runtime | `MultiAgentAssistant` multi-agent orchestration |
| Agent roles | 6, including 3 specialist workers |
| Exported tools | 45 |
| Vector collections | 3 |
| Evaluation corpus | 72 scenarios across 6 domains |
| Automation | 4 GitHub Actions workflows |
| Answer languages | European Portuguese (PT-PT) and English |

## 🗺️ Documentation Map

| Document | Focus |
|---|---|
| [Project Overview](./01_PROJECT_OVERVIEW.md) | Scope, audiences, research context, and current system snapshot |
| [System Architecture](./02_SYSTEM_ARCHITECTURE.md) | Routing, workers, conditional QA, planning, state, and providers |
| [Tools Reference](./03_TOOLS_REFERENCE.md) | The 45-tool inventory, agent ownership, coverage boundaries, and smoke checks |
| [Data Sources and Schemas](./04_DATA_SOURCES_AND_SCHEMAS.md) | Provenance, freshness, local and release artifacts, and schemas |
| [Deployment and Operations](./05_DEPLOYMENT_AND_OPERATIONS.md) | Setup, configuration, deployment, automation, validation, and troubleshooting |
| [Evaluation README](../eval/README.md) | Benchmark, ablation, judges, statistics, and interpretation boundaries |
| [Repository README](../README.md) | Public overview and main onboarding path |

## 🚀 Quick Start

Use Python **3.10 to 3.13** and configure one supported LLM provider in `.env` before launching the app:

```bash
python -m pip install -r requirements.txt
cp .env.example .env    # Windows PowerShell: Copy-Item .env.example .env
python tools/vector_store.py
streamlit run app.py
```

> [!TIP]
> When no local vector database exists, the app can download it from the configured GitHub Release at startup. For the full research environment, use `conda env create -f environment_local_gpu.yml`.

## 🧭 Recommended Reading Paths

- **Onboarding:** [Repository README](../README.md) → [Project Overview](./01_PROJECT_OVERVIEW.md) → [System Architecture](./02_SYSTEM_ARCHITECTURE.md)
- **Capabilities and provenance:** [Tools Reference](./03_TOOLS_REFERENCE.md) → [Data Sources and Schemas](./04_DATA_SOURCES_AND_SCHEMAS.md)
- **Local or hosted operation:** [Deployment and Operations](./05_DEPLOYMENT_AND_OPERATIONS.md) → [`.env.example`](../.env.example)
- **Evaluation:** [Evaluation README](../eval/README.md) → [analysis notebook](../eval/benchmark_ablation_analysis.ipynb)

## ✅ Sources of Truth

| File | Defines |
|---|---|
| [`agent/graph.py`](../agent/graph.py) and [`agent/state.py`](../agent/state.py) | Orchestration and state schema |
| [`tools/__init__.py`](../tools/__init__.py) | Exported tool registry |
| [`config.py`](../config.py) and [`.env.example`](../.env.example) | Provider, model, path, and release configuration |
| [`.github/workflows/`](../.github/workflows/) | Data refresh, vector sync, transport assets, and deployment |
| [`eval/evaluation_groundtruth_queries.json`](../eval/evaluation_groundtruth_queries.json) | Shared 72-scenario evaluation corpus |
| [`eval/run_benchmark.py`](../eval/run_benchmark.py) and [`eval/run_ablation.py`](../eval/run_ablation.py) | Evaluation subsets and execution logic |

## 📌 Documentation Boundaries

- `tools/vector_store.py` and `tools/location_resolver.py` are support infrastructure and are not counted among the 45 exported tools.
- Only the vector collections use RAG. Structured APIs, GTFS/GTFS-RT feeds, geospatial lookups, and open-data retrieval are queried directly.
- Final answers are written in PT-PT or English. Requests in other languages receive an English answer with a short language note.
- The conceptual framework figure is available as [SVG](../img/LISBOA_Framework.svg) and [PNG](../img/LISBOA_Framework.png).
