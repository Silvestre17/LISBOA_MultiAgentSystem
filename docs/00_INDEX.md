# 📚 LISBOA Documentation Hub

This documentation set describes the repository as it exists in code today. It has been reviewed against the supported Streamlit entrypoint `app.py`, the multi-agent runtime in `agent/graph.py`, the exported tool registry in `tools/__init__.py`, the GitHub Actions workflows, the strict live-test prerequisites in `tests/conftest.py`, and the evaluation notebook in `eval/benchmark_ablation_analysis.ipynb`.

> [!IMPORTANT]
> Public setup, runtime, and validation instructions support `app.py` as the documented entrypoint.
> The repository also contains `app_vprod.py` as an alternative UI variant, but it is outside the supported public documentation path.

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
| `docs/06_FUTURE_ENHANCEMENTS.md` | roadmap ideas and research directions | you are planning next iterations |
| `../README.md` | public repository overview and guided entrypoint | you are onboarding to the repo |
| `../eval/README.md` | benchmark, ablation, live coverage, and judge pipeline | you are working on evaluation |

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python tools/vector_store.py
streamlit run app.py
```

## 🧭 Recommended Reading Paths

- **New to the repository:** `../README.md` -> `docs/01_PROJECT_OVERVIEW.md` -> `docs/02_SYSTEM_ARCHITECTURE.md`
- **Need the exact capability map:** `docs/03_TOOLS_REFERENCE.md` -> `docs/04_DATA_SOURCES_AND_SCHEMAS.md`
- **Want to run locally:** `docs/05_DEPLOYMENT_AND_OPERATIONS.md` -> `../.env.example`
- **Working on evaluation:** `../eval/README.md` -> `../eval/benchmark_ablation_analysis.ipynb`
- **Need live-test readiness:** `docs/05_DEPLOYMENT_AND_OPERATIONS.md` plus `tests/conftest.py`

## ✅ Canonical References Used Across the Docs

The following files are treated as the source of truth when counts, roles, or workflows are documented:

- `tools/__init__.py` for the exported 45-tool registry
- `agent/graph.py` for the default runtime orchestration, response flow, and agent responsibilities
- `agent/state.py` for the shared `AgentState`
- `config.py` for provider defaults, agent-model mappings, and core paths
- `.github/workflows/data_pipeline.yml` and `.github/workflows/sync_vector_db.yml` for automation
- `tests/conftest.py` for strict live prerequisite enforcement
- `eval/benchmark_ablation_analysis.ipynb` for benchmark and ablation CSV exports

## 📝 Notes

- Some tool outputs and upstream payloads can contain Portuguese text because several Lisbon public sources publish in Portuguese.
- `tools/vector_store.py` is operational infrastructure and CLI support. It is important, but it is not counted as one of the 45 exported runtime tools.
- The `docs/` set intentionally focuses on the supported public application path. Auxiliary thesis material and alternative UI variants present in the repository are mentioned only when they help avoid confusion.
