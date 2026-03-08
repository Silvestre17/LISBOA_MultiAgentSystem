# ⚙️ Deployment and Operations

This guide covers local setup, environment configuration, runtime workflows, CI automation, validation, evaluation artefacts, and troubleshooting.

> [!IMPORTANT]
> The supported Streamlit launch command is `streamlit run app.py`.

## ✅ Setup Checklist

| Requirement | Needed for | Notes |
|-------------|------------|-------|
| Python 3.10+ | all local workflows | required by the repository and GitHub Actions |
| Git | cloning and updating the repository | standard prerequisite |
| One configured LLM provider | runtime assistant and evaluation | Azure OpenAI, OpenAI, or LM Studio |
| Metro credentials | full official Metro realtime experience | optional for basic fallback status checks |
| Tavily API key | web fallback and strict live coverage | optional for casual local use, required for strict live tests |

## 🔐 Environment Configuration

Start from the template:

```bash
copy .env.example .env
```

Then fill in only the providers and services you plan to use.

### Provider Selection Guide

| Provider | What you need | Best fit | Notes |
|----------|---------------|----------|-------|
| Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME` | default documented path | `config.py` currently defaults to Azure |
| OpenAI | `OPENAI_API_KEY`, optionally `OPENAI_MODEL_NAME` | simpler cloud setup | direct OpenAI API path |
| LM Studio | local server URL and model name | offline or low-cost local experimentation | no API key required |

### Runtime Environment Variables

#### LLM Providers

- `OPENAI_API_KEY`
- `OPENAI_MODEL_NAME`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT_NAME`

#### Metro Official API

- `METRO_CONSUMER_KEY`
- `METRO_CONSUMER_SECRET`

Optional Metro TLS overrides from `.env.example`:

- `METRO_CA_BUNDLE`
- `METRO_SSL_VERIFY`

#### Optional Services and Observability

- `TAVILY_API_KEY`
- `LANGCHAIN_TRACING_V2`
- `LANGCHAIN_API_KEY`
- `LANGCHAIN_PROJECT`
- `LANGCHAIN_ENDPOINT`

### Provider Behavior Notes

- The default runtime mode is multi-agent.
- Per-agent model mappings are configured in `config.py` through `AGENT_MODELS_AZURE`, `AGENT_MODELS_OPENAI`, and `AGENT_MODELS_LMSTUDIO`.
- The Streamlit sidebar in `app.py` can override the active provider and per-agent model selection at runtime.

## 🚀 First Run

```bash
pip install -r requirements.txt
python tools/vector_store.py
streamlit run app.py
```

When `app.py` starts, it also:

- pre-warms the vector store
- initializes or refreshes the Carris support database when needed
- loads environment values from `.env`

## 🧰 Vector-Store Operations

Useful commands:

```bash
python tools/vector_store.py --stats
python tools/vector_store.py --test
python tools/vector_store.py --rebuild-events
python tools/vector_store.py --rebuild-places
python tools/vector_store.py --rebuild-pdf
python tools/vector_store.py --rebuild-all
python tools/vector_store.py --no-gpu --max-docs 200
```

## ✅ Validation Ladder

### 1. Syntax and fast deterministic checks

Use this layer before slower runs:

```bash
python tests/syntax_check.py
python -m pytest eval/tests/test_benchmark_utils.py eval/tests/test_cost_accounting.py eval/tests/test_llm_judge.py eval/tests/test_validators.py -v
```

### 2. Runtime-facing regression subset

This subset exercises QA, prompt, and transport-facing paths:

```bash
python -m pytest tests/test_qa_agent.py tests/test_qa_integration.py tests/test_prompts.py tests/test_lisbon_transport.py -s -W "error::langgraph.warnings.LangGraphDeprecatedSinceV10"
```

### 3. Strict live coverage

This suite is intentionally loud about missing prerequisites:

```bash
python -m pytest tests/test_tool_prompt_coverage.py -m "live and coverage" -v
```

Strict live coverage currently validates that the following are available:

- active provider credentials for the configured LLM backend
- `METRO_CONSUMER_KEY`
- `METRO_CONSUMER_SECRET`
- `TAVILY_API_KEY`
- `data/vector_db/`
- `data_collection/webscraping/events.json`
- `data_collection/webscraping/places.json`
- `data/carris/carris.db`
- `data/cp/cp_gtfs.db`

### 4. Benchmark and ablation runs

```bash
python eval/run_benchmark.py --mode run_test
python eval/run_benchmark.py --mode full
python eval/run_benchmark.py --limit 5
python eval/run_ablation.py --mode run_test
python eval/run_ablation.py --mode full
```

For the evaluation model, refer to `eval/README.md` for judge-specific details and output schema notes.

## 📦 Evaluation Artefacts and Notebook Exports

| Artefact family | Default location | Notes |
|-----------------|------------------|-------|
| benchmark JSON outputs | `eval/results/benchmark/` | produced by `eval/run_benchmark.py` |
| ablation JSON outputs | `eval/results/ablation/` | produced by `eval/run_ablation.py` |
| strict live coverage JSON outputs | `eval/results/coverage/` | produced by live coverage runs |
| calibration JSON outputs | `eval/results/calibration/` | produced by `eval/human_calibration/run_calibration.py` |

The analysis notebook `eval/benchmark_ablation_analysis.ipynb` also exports latest CSV summaries through `flatten_benchmark_results()` and `flatten_ablation_results()`:

- `eval/results/benchmark/benchmark_flat_latest.csv`
- `eval/results/benchmark/benchmark_summary_latest.csv`
- `eval/results/ablation/ablation_flat_latest.csv`
- `eval/results/ablation/ablation_summary_latest.csv`

## 🔄 GitHub Actions Automation

| Workflow | Trigger | Purpose | Main outputs |
|----------|---------|---------|--------------|
| `data_pipeline.yml` | daily at **04:00 UTC**, plus manual trigger | scrape VisitLisboa events daily and places weekly on Mondays | updated JSON artefacts under `data_collection/webscraping/` |
| `sync_vector_db.yml` | `workflow_run` after `Update Lisbon Data`, plus manual trigger | incrementally sync ChromaDB collections and commit vector DB updates | updated artefacts under `data/vector_db/` |

### Exit-code protocol used by the Sync Workflow

| Exit code | Meaning |
|----------:|---------|
| `0` | sync complete |
| `2` | more work pending, safe to continue in another iteration |
| `143` | runner terminated the process, treated as a graceful partial stop |

## 🚦 Performance and Batching Notes

- `sync_vector_db.yml` uses batched vector-store updates to avoid CI timeouts.
- `--max-docs` limits the number of documents processed per collection in a single sync pass.
- Lower `max_docs` values reduce per-run pressure when the collection changes are large.
- The repository caches pip dependencies and Hugging Face model downloads during CI.

## 🩺 Troubleshooting

### ChromaDB Database Locked

**Typical symptom:**

```text
sqlite3.OperationalError: database is locked
```

**Typical response:**

1. stop concurrent Python processes using the vector store
2. remove stale SQLite WAL and SHM lock files if they exist
3. rerun the vector-store command once the database is free

### Missing Metro Credentials

If `METRO_CONSUMER_KEY` and `METRO_CONSUMER_SECRET` are missing, the system can still use the public fallback for some metro functionality, but the full official API experience is not available.

### Strict Live Coverage Fails Immediately

If the live suite fails before any meaningful execution, inspect the environment and the local artefacts listed in `tests/conftest.py`. The suite is designed to fail loudly rather than silently skip missing prerequisites.

### CI Sync Taking Too Long

If vector synchronization repeatedly times out in GitHub Actions:

- reduce `max_docs`
- rerun the workflow manually if the previous run exited with `2`
- inspect whether the workflow reached the iteration cap before completion

### Local Model Connectivity

If using LM Studio:

- ensure the local server is running
- confirm the base URL matches `Config.LMSTUDIO_BASE_URL`
- confirm the loaded model matches the model name expected by the runtime
