# 📚 LISBOA Documentation Hub

This documentation set describes the repository as it exists in code today. It has been verified against the Streamlit entrypoint `app.py`, the multi-agent runtime in `agent/graph.py`, the exported tool registry in `tools/__init__.py`, the GitHub Actions workflows under `.github/workflows/`, and the evaluation notebook in `eval/benchmark_ablation_analysis.ipynb`.

> [!IMPORTANT]
> The supported public entrypoint for setup, runtime, and validation is `app.py`. Run it with `streamlit run app.py`.

> [!NOTE]
> Conceptual thesis framework figure: [`../img/LISBOA_Framework.png`](../img/LISBOA_Framework.png) (vector source: [`../img/LISBOA_Framework.svg`](../img/LISBOA_Framework.svg)).

## 🔎 At a Glance

| Item | Current value |
|------|---------------|
| Supported UI entrypoint | `app.py` |
| Runtime mode | Multi-agent by default |
| Specialized agents | 6 |
| Exported LangChain tools | 45 |
| Evaluation corpus | 72 benchmark queries |
| Vector collections | 3 |
| Core workflows | `data_pipeline.yml`, `sync_vector_db.yml` |

## 🗺️ Documentation Map

| Document | What it covers | Best used when |
|----------|----------------|----------------|
| `docs/01_PROJECT_OVERVIEW.md` | scope, audiences, project snapshot, repository boundaries | you need the big picture fast |
| `docs/02_SYSTEM_ARCHITECTURE.md` | orchestration, routing, QA, planning, state, providers | you want to understand runtime behavior |
| `docs/03_TOOLS_REFERENCE.md` | authoritative tool inventory, agent-to-tool mapping, smoke checks | you need exact capabilities by domain |
| `docs/04_DATA_SOURCES_AND_SCHEMAS.md` | source systems, refresh cadence, artefacts, vector collections | you are checking freshness, provenance, or storage |
| `docs/05_DEPLOYMENT_AND_OPERATIONS.md` | setup, `.env` configuration, CI workflows, validation, troubleshooting | you want to run, test, or operate the project |
| `../README.md` | public repository overview and guided entrypoint | you are onboarding to the repo |
| `../eval/README.md` | benchmark, ablation, live coverage, and judge pipeline | you are working on evaluation |

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python tools/vector_store.py
streamlit run app.py
```

> [!TIP]
> Need the full local environment (scraping, evaluation, notebooks, CUDA-enabled PyTorch)? Use `conda env create -f environment_local_gpu.yml` instead. See [`05_DEPLOYMENT_AND_OPERATIONS.md`](./05_DEPLOYMENT_AND_OPERATIONS.md) for details.

## 🧭 Recommended Reading Paths

- **New to the repository:** [`../README.md`](../README.md) → [`01_PROJECT_OVERVIEW.md`](./01_PROJECT_OVERVIEW.md) → [`02_SYSTEM_ARCHITECTURE.md`](./02_SYSTEM_ARCHITECTURE.md)
- **Need the exact capability map:** [`03_TOOLS_REFERENCE.md`](./03_TOOLS_REFERENCE.md) → [`04_DATA_SOURCES_AND_SCHEMAS.md`](./04_DATA_SOURCES_AND_SCHEMAS.md)
- **Want to run it locally:** [`05_DEPLOYMENT_AND_OPERATIONS.md`](./05_DEPLOYMENT_AND_OPERATIONS.md) → [`../.env.example`](../.env.example)
- **Working on evaluation:** [`../eval/README.md`](../eval/README.md) → [`../eval/benchmark_ablation_analysis.ipynb`](../eval/benchmark_ablation_analysis.ipynb)

## ✅ Canonical Sources of Truth

When counts, roles, or workflows are documented, these files are the reference:

- [`tools/__init__.py`](../tools/__init__.py) — exported 45-tool registry
- [`agent/graph.py`](../agent/graph.py) — orchestration, routing, response flow
- [`agent/state.py`](../agent/state.py) — shared `AgentState` schema
- [`config.py`](../config.py) — provider defaults, agent-model mappings, paths
- [`.github/workflows/data_pipeline.yml`](../.github/workflows/data_pipeline.yml) and [`.github/workflows/sync_vector_db.yml`](../.github/workflows/sync_vector_db.yml) — automation
- [`eval/evaluation_groundtruth_queries.json`](../eval/evaluation_groundtruth_queries.json) — 72-query evaluation corpus
- [`eval/benchmark_ablation_analysis.ipynb`](../eval/benchmark_ablation_analysis.ipynb) — benchmark and ablation CSV exports

## 📝 Notes

> [!NOTE]
> Upstream payloads are often in Portuguese because Lisbon public sources publish in Portuguese. Final user-facing answers are emitted only in **PT-PT** or **English**; other input languages receive an English answer with a short bilingual note.

- `tools/vector_store.py` is operational infrastructure and is **not** counted among the 45 exported runtime tools.
- The `docs/` set covers the supported public path centred on `app.py`. Auxiliary thesis material may live in the repository, but it is not part of the public operating path unless explicitly stated.
