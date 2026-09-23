# ⚙️ Deployment and Operations

This guide covers local setup, environment configuration, Docker and hosted deployment, vector-store operations, validation, automation, and troubleshooting.

> [!IMPORTANT]
> The supported launch command is `streamlit run app.py`.

## ✅ Setup Checklist

| Requirement | Needed For | Notes |
|---|---|---|
| Python 3.10 to 3.13 | All local workflows | Declared in `pyproject.toml` |
| Git | Cloning and updating the repository | Standard prerequisite |
| One configured LLM provider | The assistant and the evaluation | Azure OpenAI, OpenAI, or LM Studio |
| Metro de Lisboa credentials | Official real-time Metro data | Optional; the public status fallback remains available |
| Tavily API key | Web fallback for history and culture questions | Optional |
| Hugging Face token | Reliable model downloads, for example `BAAI/bge-m3` | Optional for public models; recommended for hosted deployments |
| LangSmith account | Tracing and monitoring | Optional |

## 🔐 Environment Configuration

Start from the template and fill in only the services you plan to use:

```bash
cp .env.example .env    # Windows PowerShell: Copy-Item .env.example .env
```

### Provider Selection

| Provider | Variables | Best Fit | Notes |
|---|---|---|---|
| Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME` | Default path | `config.py` defaults to Azure and uses the v1 API, so no API version is needed |
| OpenAI | `OPENAI_API_KEY`, optionally `OPENAI_MODEL_NAME` | Simplest cloud setup | Direct OpenAI API |
| LM Studio | `LMSTUDIO_BASE_URL` and `LMSTUDIO_MODEL_NAME` in `config.py` | Offline or low-cost experimentation | No API key; start the local server on port 1234 |

To switch provider, change `Config.MODEL_PROVIDER` in `config.py`. Per-agent models are set in `AGENT_MODELS_AZURE`, `AGENT_MODELS_OPENAI`, and `AGENT_MODELS_LMSTUDIO`.

### Metro de Lisboa API

| Variable | Purpose |
|---|---|
| `METRO_CONSUMER_KEY`, `METRO_CONSUMER_SECRET` | OAuth2 credentials from the Metro de Lisboa API Store (`EstadoServicoML` subscription) |
| `METRO_CA_BUNDLE` | Optional custom trust bundle (PEM) |
| `METRO_SSL_VERIFY` | Optional; `false` disables certificate verification and is meant only for local diagnosis |
| `METRO_SSL_ALLOW_INSECURE_FALLBACK` | Optional; `true` allows one insecure retry after secure validation and dynamic chain completion both fail |

Certificate verification is enabled by default. When the Metro gateway serves an incomplete certificate chain, the code builds the missing issuer chain from the live certificate's AIA metadata and retries securely, so no PEM file is needed in the repository.

> [!CAUTION]
> The insecure TLS fallback is **not recommended for deployed environments**. Use it only as a temporary diagnostic measure.

### Runtime Data and Startup

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Read-only Hugging Face token for model downloads |
| `VECTOR_DB_RELEASE_ENABLED`, `VECTOR_DB_RELEASE_REPO`, `VECTOR_DB_RELEASE_TAG`, `VECTOR_DB_RELEASE_ASSET` | Download `vector_db.zip` from a GitHub Release when no local vector database exists |
| `VECTOR_DB_RELEASE_TOKEN` | Read-only GitHub token, needed only if the repository is private |
| `TRANSPORT_DATA_RELEASE_REPO`, `TRANSPORT_DATA_RELEASE_TAG` | Release that holds the transport runtime assets |
| `CARRIS_RUNTIME_RELEASE_ENABLED`, `CARRIS_RUNTIME_RELEASE_ASSET`, `CP_RUNTIME_RELEASE_ENABLED`, `CP_RUNTIME_RELEASE_ASSET` | Last-known-good Carris Urban and CP SQLite files, used when a live GTFS download fails |
| `LISBOA_RUNTIME_DATA_DIR`, `VECTOR_DB_DIR` | Optional writable locations for generated data in hosted runtimes |
| `LISBOA_REUSE_LOCAL_TRANSPORT_DATA` | Local smoke-test acceleration; ignored on Hugging Face Spaces |
| `STREAMLIT_RESOURCE_CACHE_TTL_SECONDS` | Optional refresh interval for release-backed resources without a restart |
| `LISBOA_STARTUP_PRELOAD_ENABLED`, `LISBOA_STARTUP_PRELOAD_REQUIRED`, `LISBOA_STARTUP_PRELOAD_LANGUAGE` | Startup preload in the Docker entry point |

### Observability and Logging

| Variable | Purpose |
|---|---|
| `TAVILY_API_KEY` | Web search for the web fallback |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT` | LangSmith tracing; use the EU endpoint for EU accounts |
| `LANGSMITH_WORKSPACE_ID` | Required when the API key is linked to several workspaces |
| `LANGSMITH_SYNC_FLUSH` | Waits for trace ingestion after each run, for debugging only; it adds latency |
| `SHOW_MARKDOWN_RESPONSE_IN_TERMINAL`, `SHOW_DETAILED_EXECUTION_LOGS` | Terminal output of final answers and execution details |

Legacy `LANGCHAIN_*` tracing aliases are still accepted, but new setups should use the `LANGSMITH_*` names.

### Tracing Behavior

- Each user request produces one top-level LangSmith trace, with nested spans for the supervisor, the workers, model calls, and tool executions.
- Model connection checks, including the **Connect System** button when credential inputs are enabled, use raw HTTP requests and create no traces.
- If the API key is linked to several workspaces and `LANGSMITH_WORKSPACE_ID` is missing, the tracing preflight check may disable tracing automatically.

## 🚀 First Run

For the Streamlit runtime:

```bash
python -m pip install -r requirements.txt
python tools/vector_store.py
streamlit run app.py
```

For the full local environment, with scraping, tests, evaluation, notebooks, and CUDA-enabled PyTorch on NVIDIA GPUs:

```bash
conda env create -f environment_local_gpu.yml
conda activate lisboa_thesis2026
python tools/vector_store.py
streamlit run app.py
```

In an existing environment, the extra research packages can be installed with:

```bash
python -m pip install -r requirements_all.txt
```

At startup, `app.py` loads `.env`, warms the Carris Urban database, the Metro station cache, the CP GTFS and AML station data, and the Carris Metropolitana caches, and pre-warms the vector store.

> [!TIP]
> The first start takes longer because `BAAI/bge-m3` is downloaded and the caches are warmed. Later starts are faster.

## 🐳 Docker and Hosted Deployment

### Local Docker

```bash
docker build -t lisboa .
docker run --env-file .env -p 8501:8501 lisboa
```

The container starts `scripts/hf_space_entrypoint.py`, which runs the startup preload and then launches Streamlit on port 8501 (override with `PORT` or `STREAMLIT_SERVER_PORT`).

### Hugging Face Spaces

`deploy_huggingface_space.yml` builds a deployment bundle with the Dockerfile, the application code, and a Space README, and uploads it to the Space `AndreSilvestre17/lisboa`:

- It requires an `HF_TOKEN` secret in the GitHub repository.
- The Space is created as a private Docker Space on the first deployment.
- The bundle includes the pricing metadata and the VisitLisboa and Lisboa Aberta source files, but not the vector database or the transport SQLite files. The Space downloads those from GitHub Releases at startup, into writable runtime storage.

## 🧰 Vector-Store Operations

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

- The JSON source files remain the source of truth.
- Checkpoints under `data/vector_db/_sync_state/` store the collection name, a semantic fingerprint of the source, the sync mode, and the document IDs still waiting for embeddings.
- If the source JSON changes before the pending queue finishes, the checkpoint is invalidated and recomputed from the new payload.
- Changed records are updated with batched upserts, so the live collection is never mass-deleted before the replacement embeddings are ready.
- Rebuild flags clear the corresponding checkpoint before rebuilding.

## ✅ Validation Ladder

> [!TIP]
> Run these in order and move to the next step only when the faster checks pass. On Windows consoles, add `-X utf8` if emoji or accented characters fail to print.

### 1. Syntax

```bash
python scripts/syntax_check.py
```

### 2. Prompt Smoke Runs

Recommended for any change to agents, prompts, formatters, the planner, QA, or routing:

```bash
python scripts/run_prompts.py --suite smoke
python scripts/run_prompts.py --prompt "How do I get from Baixa-Chiado to Aeroporto?" --language en --quiet
```

Run at least one prompt and one variant with a different entity, language, or wording.

### 3. Transport Verification

```bash
python scripts/run_transport_verification.py
```

### 4. Provider Consistency

```bash
python scripts/run_provider_consistency.py
```

### 5. Benchmark and Ablation

```bash
python -m eval.run_benchmark --mode run_test
python -m eval.run_benchmark --mode full
python -m eval.run_benchmark --limit 5
python -m eval.run_ablation --mode run_test
python -m eval.run_ablation --mode full
```

> [!IMPORTANT]
> The benchmark and ablation runners must be invoked as modules (`python -m eval.run_benchmark`); running the scripts directly breaks the `agent` imports.

The judge details and output schemas are in the [Evaluation README](../eval/README.md).

## 📦 Evaluation Artifacts

| Artifact | Default Location | Produced By |
|---|---|---|
| Benchmark JSON outputs | `eval/results/benchmark/` | `eval/run_benchmark.py` |
| Ablation JSON outputs | `eval/results/ablation/` | `eval/run_ablation.py` |
| Statistical analysis (JSON and CSV) | `eval/results/statistics/` | `eval/statistical_analysis.py` |
| Figures | `eval/results/figures/` | The analysis notebook |

The analysis notebook `eval/benchmark_ablation_analysis.ipynb` also exports the latest CSV summaries through `flatten_benchmark_results()` and `flatten_ablation_results()`:

- `eval/results/benchmark/benchmark_flat_latest.csv`
- `eval/results/benchmark/benchmark_summary_latest.csv`
- `eval/results/ablation/ablation_flat_latest.csv`
- `eval/results/ablation/ablation_summary_latest.csv`

## 🔄 GitHub Actions Automation

| Workflow | Trigger | Purpose | Main Output |
|---|---|---|---|
| `data_pipeline.yml` | Daily at **04:00 UTC**; manual runs with `events`, `places`, or `both` | Scrape VisitLisboa events daily and places on Mondays | JSON artifacts committed under `data_collection/webscraping/` |
| `sync_vector_db.yml` | After a successful `Update Lisbon Data` run; manual runs | Incremental ChromaDB sync in bounded batches | `vector_db.zip` (complete) or `vector_db_staging.zip` (in progress) on the `vector-db-latest` release |
| `sync_transport_runtime_data.yml` | Daily at **03:25 UTC**; pushes that change the transport release code; manual runs | Refresh the Carris Urban and CP runtime assets | Fixed-name ZIP files and a manifest on the `transport-data-latest` release |
| `deploy_huggingface_space.yml` | Pushes to `main` that change the app; completed vector or transport syncs; manual runs | Deploy the app | Updated Hugging Face Space |

GitHub Actions schedules run in UTC, so 04:00 UTC is 05:00 in Lisbon during summer time.

### Vector Sync Protocol

| Exit Code | Meaning |
|---:|---|
| `0` | Sync complete; the complete asset is published |
| `2` | More work pending; a staging asset is published and the next iteration continues |
| `143` | The runner stopped the process; a staging asset is published and a new run resumes from it |

- Each run restores the staging asset first, falling back to the latest complete asset.
- A run processes up to 10 batches of 200 documents by default (`max_docs` can be changed in manual runs).
- Runs are serialized per branch, and the job timeout stays below GitHub's six-hour limit.
- Events are synchronized before places, so time-sensitive updates arrive first.
- pip packages and Hugging Face models are cached between runs.

## 🩺 Troubleshooting

### ChromaDB Database Locked

```text
sqlite3.OperationalError: database is locked
```

1. Stop other Python processes that use the vector store.
2. Remove stale SQLite WAL and SHM lock files, if any.
3. Run the vector-store command again once the database is free.

### Missing Metro Credentials

Without `METRO_CONSUMER_KEY` and `METRO_CONSUMER_SECRET`, the public fallback still covers part of the Metro functionality, but the official real-time data are not available.

### Metro TLS Fails with Valid Credentials

The expected sequence is:

1. Normal certificate verification.
2. Automatic completion of missing issuer certificates.
3. An optional insecure retry, only when `METRO_SSL_ALLOW_INSECURE_FALLBACK=true`.

> [!WARNING]
> If a deployed environment still fails after step 2, keep `METRO_SSL_VERIFY=true` and inspect the outbound network policy or any TLS interception.

### A Live Integration Seems Broken

Run the corresponding tool module directly (for example, `python tools/ipma_api.py`), check the credentials in `.env`, and then run the prompt smoke suite or the operator-specific scripts under `scripts/`.

### Vector Sync Takes Too Long in CI

- Lower `max_docs` in a manual run.
- Run `python tools/vector_store.py --stats --no-gpu` locally to check for pending sync work.
- Rerun the workflow manually if the previous run ended with exit code `2` or `143`.
- Check whether the run reached the 10-iteration cap.
- Inspect `data/vector_db/_sync_state/` only for diagnosis, and delete a checkpoint only if the source JSON changed and the saved queue is clearly stale or corrupted.

### LM Studio Connectivity

- Make sure the local server is running.
- Check that the base URL matches `Config.LMSTUDIO_BASE_URL`.
- Check that the loaded model matches the model name the runtime expects.
