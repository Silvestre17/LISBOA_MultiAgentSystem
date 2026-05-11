# ⚙️ Deployment and Operations

This guide covers local setup, environment configuration, runtime workflows, CI automation, validation, evaluation artefacts, and troubleshooting.

> [!IMPORTANT]
> The supported Streamlit launch command is `streamlit run app.py`.

## ✅ Setup Checklist

| Requirement | Needed for | Notes |
|-------------|------------|-------|
| Python 3.10+ | All local workflows | Required by the repository and GitHub Actions |
| Git | Cloning and updating the repository | Standard prerequisite |
| One configured LLM provider | Runtime assistant and evaluation | Azure OpenAI, OpenAI, or LM Studio |
| Metro credentials | Full official Metro realtime experience | Optional; fallback status remains available |
| Tavily API key | Web fallback for history and culture queries | Optional |

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

Metro TLS notes:

- By default, the runtime keeps certificate verification enabled.
- If the Metro gateway serves an incomplete TLS chain, the code builds a temporary CA bundle dynamically from the live certificate's AIA issuer chain and retries securely.
- No repository PEM file is required for this default path.
- `METRO_CA_BUNDLE` is only for explicit custom trust bundles.
- `METRO_SSL_VERIFY=false` disables verification outright and should be limited to local diagnosis.
- `METRO_SSL_ALLOW_INSECURE_FALLBACK=true` allows one insecure retry only after secure validation and dynamic chain completion both fail.

> [!CAUTION]
> Insecure TLS fallback is **not recommended for deployed environments**. Use it only as a temporary diagnostic measure.

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

- Runtime mode is **multi-agent** through `MultiAgentAssistant`.
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

- Warms the Carris Urban support database, Metro station cache, CP GTFS plus AML station support data, and Carris Metropolitana caches.
- Pre-warms the vector store for the multi-agent knowledge layer.
- Loads environment values from `.env`.

> [!TIP]
> First boot may take longer because of model downloads (`BAAI/bge-m3`) and cache warmup. Subsequent runs are noticeably faster.

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

Resumable sync behaviour:

- JSON source files remain the source of truth.
- The sync process persists only checkpoint metadata under `data/vector_db/_sync_state/`.
- Each checkpoint stores the collection name, semantic source fingerprint, sync mode, and pending document IDs that still need embedding.
- If the source JSON changes before the pending queue finishes, the checkpoint is invalidated automatically and recomputed from the fresh JSON payload.
- Modified records are updated with batched upserts, so the live collection is not mass-deleted before the replacement embeddings are ready.
- Rebuild flags clear the corresponding checkpoint before rebuilding.

## ✅ Validation Ladder

> [!TIP]
> Run these in order, escalating only when faster checks pass.

### 1. Syntax and dataset integrity (fast)

```bash
python scripts/syntax_check.py
python -m pytest eval/tests/ -q
```

`eval/tests/` currently protects deterministic dataset shape and validator helpers (`test_dataset_integrity.py`, `test_validators.py`). It is intentionally lean and not the main proof of user-facing answer quality.

### 2. Prompt smoke runs (recommended for prompt/agent changes)

```bash
python scripts/run_prompts.py --suite smoke
python scripts/run_prompts.py --prompt "How do I get from Baixa-Chiado to Aeroporto?" --language en --quiet
```

For any change to agents, prompts, formatters, planner, QA, or routing logic, run at least one prompt plus one variant (different entity, language, or wording).

### 3. Transport-specific verification

```bash
python scripts/run_transport_verification.py
```

### 4. Provider consistency

```bash
python scripts/run_provider_consistency.py
```

### 5. Benchmark and ablation runs (research)

```bash
python -m eval.run_benchmark --mode run_test
python -m eval.run_benchmark --mode full
python -m eval.run_benchmark --limit 5
python -m eval.run_ablation  --mode run_test
python -m eval.run_ablation  --mode full
```

> [!IMPORTANT]
> Benchmark and ablation runners must be invoked in module form (`python -m eval.run_benchmark`). Direct script invocation breaks `agent` import resolution.

For judge-specific details and the output schema, refer to [`eval/README.md`](../eval/README.md).

## 📦 Evaluation Artefacts and Notebook Exports

| Artefact family | Default location | Notes |
|-----------------|------------------|-------|
| Benchmark JSON outputs | `eval/results/benchmark/` | Produced by `eval/run_benchmark.py` |
| Ablation JSON outputs | `eval/results/ablation/` | Produced by `eval/run_ablation.py` |
| Statistical analysis JSON/CSV outputs | `eval/results/statistics/` | Produced by `eval/statistical_analysis.py` |
| Figures | `eval/results/figures/` | Produced by the analysis notebook |

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
- The workflow timeout is configured below the GitHub-hosted 6-hour hard job limit, while still leaving room for dependency installation and final repository operations.

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

1. Stop concurrent Python processes using the vector store.
2. Remove stale SQLite WAL and SHM lock files if they exist.
3. Rerun the vector-store command once the database is free.

### Missing Metro Credentials

If `METRO_CONSUMER_KEY` and `METRO_CONSUMER_SECRET` are missing, the system can still use the public fallback for some metro functionality, but the full official API experience is not available.

### Metro TLS Chain Fails Even With Valid Credentials

Preferred behaviour:

1. Normal certificate verification.
2. Automatic dynamic completion of missing issuer certificates.
3. Optional insecure retry only when `METRO_SSL_ALLOW_INSECURE_FALLBACK=true`.

> [!WARNING]
> If a deployed environment still fails after step 2, keep `METRO_SSL_VERIFY=true` and inspect outbound network policy or TLS interception. Use insecure fallback only as a temporary diagnostic measure.

### Strict Live Coverage Fails Immediately

The legacy strict-live-coverage suite under `tests/` was retired during the 2026-05 cleanup. Per-tool live coverage is now exercised through real prompt smoke runs (`scripts/run_prompts.py`) and the operator-specific verification scripts under `scripts/`. If a live integration appears broken, run the targeted tool module directly (for example `python tools/ipma_api.py`) and check provider credentials in `.env`.

### CI Sync Taking Too Long

If vector synchronization repeatedly times out in GitHub Actions:

- Reduce `max_docs`.
- Inspect `python tools/vector_store.py --stats --no-gpu` to see whether any collection reports pending sync work.
- Rerun the workflow manually if the previous run exited with `2`.
- Inspect whether the workflow reached the iteration cap before completion.
- Inspect `data/vector_db/_sync_state/` only when diagnosis is required, and delete a checkpoint manually only if the source JSON changed and the saved queue is demonstrably stale or corrupted.

### Local Model Connectivity

If using LM Studio:

- Ensure the local server is running.
- Confirm the base URL matches `Config.LMSTUDIO_BASE_URL`.
- Confirm the loaded model matches the model name expected by the runtime.
