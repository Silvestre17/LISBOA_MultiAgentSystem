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
- `METRO_SSL_ALLOW_INSECURE_FALLBACK`

Metro TLS note:

- By default, the runtime keeps certificate verification enabled.
- If the Metro gateway serves an incomplete TLS chain, the code now builds a temporary CA bundle dynamically from the live certificate's AIA issuer chain and retries securely.
- No repository PEM file is required for this default path.
- `METRO_CA_BUNDLE` is only for explicit custom trust bundles.
- `METRO_SSL_VERIFY=false` disables verification outright and should be limited to local diagnosis.
- `METRO_SSL_ALLOW_INSECURE_FALLBACK=true` allows one insecure retry only after secure validation and dynamic chain completion both fail. This is not recommended for deployed environments.

#### Optional Services and Observability

- `TAVILY_API_KEY`
- `LANGSMITH_TRACING`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `LANGSMITH_ENDPOINT`
- `LANGSMITH_WORKSPACE_ID` when the LangSmith API key is linked to multiple workspaces
- `LANGSMITH_SYNC_FLUSH` to force a post-run tracer flush and `read_run` confirmation probe when tracing persistence needs active debugging

Legacy `LANGCHAIN_*` tracing aliases are still accepted by the runtime for backward compatibility, but the canonical `LANGSMITH_*` names above should be preferred for new setups.

### Provider Behavior Notes

- Default runtime mode is **multi-agent** (`Config.USE_MULTI_AGENT = True`).
- Per-agent model mappings: `AGENT_MODELS_AZURE`, `AGENT_MODELS_OPENAI`, `AGENT_MODELS_LMSTUDIO` in `config.py`.
- The Streamlit sidebar can override active provider and per-agent model at runtime.
- UI provider connection tests use raw HTTP requests (not LangChain), so health checks do **not** create LangSmith traces.

### LangSmith Tracing Notes

- A real user request should produce exactly one top-level LangSmith trace.
- Nested spans then capture the supervisor, worker agents, LangChain model calls, and tool executions used to answer that request.
- The connection-validation flow behind `Save & Connect` in `app.py` is intentionally excluded from tracing to avoid wasting the free-tier trace quota.
- If your LangSmith API key is linked to multiple workspaces, set `LANGSMITH_WORKSPACE_ID` or the tracing preflight check may auto-disable tracing.
- `LANGSMITH_SYNC_FLUSH=true` makes the runtime wait for the local tracer queue to flush and then attempt a `read_run` confirmation after each user-facing run. The default remains off so normal chat latency is not increased.

## 🚀 First Run

For the production-style Streamlit runtime:

```bash
pip install -r requirements.txt
python tools/vector_store.py
streamlit run app.py
```

For the full local environment with scraping, tests, evaluation, notebooks,
and CUDA-enabled PyTorch on NVIDIA systems:

```bash
conda env create -f environment_local_gpu.yml
conda activate lisboa_thesis2026
python tools/vector_store.py
streamlit run app.py
```

If you already created the environment manually, install the extra local-only
packages with:

```bash
pip install -r requirements_all.txt
```

When `app.py` starts, it also:

- warms the Carris Urban support database, Metro station cache, CP GTFS plus AML station support data, and Carris Metropolitana caches
- pre-warms the vector store only when multi-agent mode is enabled
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

Resumable sync behavior:

- JSON source files remain the source of truth.
- The sync process persists only checkpoint metadata under `data/vector_db/_sync_state/`.
- Each checkpoint stores the collection name, semantic source fingerprint, sync mode, and pending document IDs that still need embedding.
- If the source JSON changes before the pending queue finishes, the checkpoint is invalidated automatically and recomputed from the fresh JSON payload.
- Modified records are updated with batched upserts, so the live collection is not mass-deleted before the replacement embeddings are ready.
- Rebuild flags clear the corresponding checkpoint before rebuilding.

## ✅ Validation Ladder

### 1. Syntax and fast deterministic checks

Use this layer before slower runs:

```bash
python scripts/syntax_check.py
python -m pytest eval/tests/test_dataset_integrity.py eval/tests/test_benchmark_utils.py eval/tests/test_cost_accounting.py eval/tests/test_llm_judge.py eval/tests/test_validators.py -v
```

### 2. Runtime-facing regression subset

This subset exercises QA, prompt, and transport-facing paths:

```bash
python -m pytest tests/test_qa_agent.py tests/test_audit_fixes.py tests/test_response_guardrails.py tests/test_transport_parity_and_rendering.py tests/test_langsmith_tracing.py tests/test_metro_api_fallback_messaging.py -q
python scripts/run_prompts.py --suite smoke
```

### 3. Strict live coverage

This suite is intentionally loud about missing prerequisites:

```bash
python -m pytest tests/test_tool_prompt_coverage.py --run-live -m "live and coverage" -v
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
python -m eval.run_benchmark --mode run_test
python -m eval.run_benchmark --mode full
python -m eval.run_benchmark --limit 5
python -m eval.run_ablation --mode run_test
python -m eval.run_ablation --mode full
```

For the evaluation model, refer to `eval/README.md` for judge-specific details and output schema notes.

## 📦 Evaluation Artefacts and Notebook Exports

| Artefact family | Default location | Notes |
|-----------------|------------------|-------|
| benchmark JSON outputs | `eval/results/benchmark/` | produced by `eval/run_benchmark.py` |
| ablation JSON outputs | `eval/results/ablation/` | produced by `eval/run_ablation.py` |
| strict live coverage JSON outputs | `eval/results/coverage/` | produced by live coverage runs |
| statistical analysis JSON/CSV outputs | `eval/results/statistics/` | produced by `eval/statistical_analysis.py` |

The analysis notebook `eval/benchmark_ablation_analysis.ipynb` also exports latest CSV summaries through `flatten_benchmark_results()` and `flatten_ablation_results()`:

- `eval/results/benchmark/benchmark_flat_latest.csv`
- `eval/results/benchmark/benchmark_summary_latest.csv`
- `eval/results/ablation/ablation_flat_latest.csv`
- `eval/results/ablation/ablation_summary_latest.csv`

## 🔄 GitHub Actions Automation

| Workflow | Trigger | Purpose | Main outputs |
|----------|---------|---------|--------------|
| `data_pipeline.yml` | daily at **04:00 UTC**, plus manual trigger with `events` / `places` / `both` | scrape VisitLisboa events daily and places weekly on Mondays, while manual runs can target either dataset or both | updated JSON artefacts under `data_collection/webscraping/` |
| `sync_vector_db.yml` | `workflow_run` after `Update Lisbon Data`, plus manual trigger | incrementally sync ChromaDB collections, persist pending checkpoints, and commit durable vector DB progress after each sync iteration | updated artefacts under `data/vector_db/`, including `_sync_state/` when work remains |

### Exit-code protocol used by the Sync Workflow

| Exit code | Meaning |
|----------:|---------|
| `0` | sync complete |
| `2` | more work pending, safe to continue in another iteration |
| `143` | runner terminated the process, treated as a graceful partial stop |

Checkpoint semantics used by the sync workflow:

- `sync_vector_db.yml` runs when scraped JSON changed and also when `_sync_state/` already contains pending work from an earlier run.
- Each sync iteration stages and pushes `data/vector_db/` immediately after the Python sync command returns, so completed progress is durable before the next iteration starts.
- Workflow concurrency is serialized per ref to avoid overlapping vector DB pushes.
- As of 2026-04, the workflow timeout is configured below the GitHub-hosted 6-hour hard job limit, while still leaving room for dependency installation and final repository operations.

## 🚦 Performance and Batching Notes

- `sync_vector_db.yml` uses batched vector-store updates to avoid CI timeouts.
- `--max-docs` limits the number of documents processed per collection in a single sync pass.
- Lower `max_docs` values reduce per-run pressure when the collection changes are large.
- The repository caches pip dependencies and Hugging Face model downloads during CI.
- Event sync runs before places sync inside the Python orchestration so time-sensitive event updates are refreshed earlier in a constrained CI window.

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

### Metro TLS Chain Fails Even With Valid Credentials

As of 2026-04, the preferred behavior is:

1. normal certificate verification
2. automatic dynamic completion of missing issuer certificates
3. optional insecure retry only when `METRO_SSL_ALLOW_INSECURE_FALLBACK=true`

If a deployed environment still fails after step 2, first keep `METRO_SSL_VERIFY=true` and inspect outbound network policy or TLS interception. Only use insecure fallback as a temporary diagnostic measure.

### Strict Live Coverage Fails Immediately

If the live suite fails before any meaningful execution, inspect the environment and the local artefacts listed in `tests/conftest.py`. The suite is designed to fail loudly rather than silently skip missing prerequisites.

### CI Sync Taking Too Long

If vector synchronization repeatedly times out in GitHub Actions:

- reduce `max_docs`
- inspect `python tools/vector_store.py --stats --no-gpu` to see whether any collection reports pending sync work
- rerun the workflow manually if the previous run exited with `2`
- inspect whether the workflow reached the iteration cap before completion
- inspect `data/vector_db/_sync_state/` only when diagnosis is required, and delete a checkpoint manually only if the source JSON changed and the saved queue is demonstrably stale or corrupted

### Local Model Connectivity

If using LM Studio:

- ensure the local server is running
- confirm the base URL matches `Config.LMSTUDIO_BASE_URL`
- confirm the loaded model matches the model name expected by the runtime
