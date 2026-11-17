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
| Hugging Face token | Reliable model downloads, for example `BAAI/bge-m3` | Optional; used only by the local embeddings backend |
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

### Visitor Model Settings

The server provider above is the default. **⚙️ System Settings** in the sidebar also lets each visitor use their own OpenAI-compatible endpoint (OpenAI, Azure OpenAI v1, Gemini, OpenRouter, a local LM Studio, and others) with their own key and model.

| Item | Behaviour |
|---|---|
| Scope | The visitor's browser session only; other sessions keep their own model |
| Key handling | Kept in `st.session_state` and passed straight to the chat-model client; never written to the environment, `Config`, files, logs, or traces |
| Check on save | One tool-calling request with tracing disabled; the result is shown as a localized message |
| Model requirement | Tool calling; the same model serves every agent |
| No server key | The app starts and asks the visitor to add a provider |
| `LISBOA_ALLOW_VISITOR_LLM_SETTINGS` | `true` by default; `false` hides the settings |
| `LISBOA_ALLOW_PRIVATE_LLM_HOSTS` | `true` locally, `false` on [hosted deployments](#streamlit-community-cloud); when `false`, base URLs that resolve to loopback, private, link-local, reserved or multicast addresses are refused, the LM Studio preset is hidden, and every request re-resolves the host, connects only to the public addresses it just checked (no DNS rebinding) and does not follow redirects |

If the server key stops working, remove it from the deployment secrets so that visitors see the setup prompt instead of failed requests.

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
| `LISBOA_HOSTED` | `true` or `false` overrides the hosted detection (Hugging Face `SPACE_ID`, or the Community Cloud checkout under `/mount/src`) |
| `LISBOA_EMBEDDINGS_BACKEND` | `local`, `cloudflare`, or `auto` (default: `cloudflare` when both Cloudflare values are set, otherwise `local`); the startup log names the active backend |
| `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` | Workers AI query embeddings (`@cf/baai/bge-m3`, L2-normalised like the store); the token needs only the "Workers AI: Read" permission |
| `HF_TOKEN` | Read-only Hugging Face token for model downloads (local backend) |
| `VECTOR_DB_RELEASE_ENABLED`, `VECTOR_DB_RELEASE_REPO`, `VECTOR_DB_RELEASE_TAG`, `VECTOR_DB_RELEASE_ASSET` | Download `vector_db.zip` from a GitHub Release when no local vector database exists |
| `VECTOR_DB_RELEASE_TOKEN` | GitHub token for release downloads and freshness checks; needed if the repository is private, recommended on Community Cloud, whose apps share the unauthenticated limit of 60 API calls per hour per IP |
| `TRANSPORT_DATA_RELEASE_REPO`, `TRANSPORT_DATA_RELEASE_TAG` | Release that holds the transport runtime assets |
| `CARRIS_RUNTIME_RELEASE_ENABLED`, `CARRIS_RUNTIME_RELEASE_ASSET`, `CP_RUNTIME_RELEASE_ENABLED`, `CP_RUNTIME_RELEASE_ASSET` | Last-known-good Carris Urban and CP SQLite files, used when a live GTFS download fails |
| `LISBOA_RUNTIME_DATA_DIR`, `VECTOR_DB_DIR` | Writable locations for generated data; default `data/` locally and `<system temp>/lisboa_runtime` on hosted deployments |
| `LISBOA_DATA_REFRESH_HOURS` | Hours between background data refresh checks on hosted deployments (default 6) |
| `LISBOA_REUSE_LOCAL_TRANSPORT_DATA` | Local smoke-test acceleration; ignored on hosted deployments |
| `LISBOA_STARTUP_PRELOAD_ENABLED`, `LISBOA_STARTUP_PRELOAD_REQUIRED`, `LISBOA_STARTUP_PRELOAD_LANGUAGE` | Startup preload in the Docker entry point |
| `IPMA_BASE_URL` | Host that serves the IPMA `/open-data/` paths; unset means `https://api.ipma.pt`. Set it to the [IPMA relay](#ipma-relay) on hosts whose network IPMA does not accept |

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
- The visitor model check (**Save and test** in the sidebar) runs with tracing disabled and creates no traces.
- If the API key is linked to several workspaces and `LANGSMITH_WORKSPACE_ID` is missing, the tracing preflight check may disable tracing automatically.

## 🚀 First Run

For the Streamlit runtime with the local embedding model (`requirements.txt` alone is the Community Cloud runtime, without torch):

```bash
python -m pip install -r requirements_local_embeddings.txt
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
> The first start takes longer because `BAAI/bge-m3` is downloaded (local backend) and the caches are warmed. Later starts are faster.

## 🐳 Docker and Hosted Deployment

### Local Docker

```bash
docker build -t lisboa .
docker run --env-file .env -p 8501:8501 lisboa
```

The container starts `scripts/hf_space_entrypoint.py`, which runs the startup preload and then launches Streamlit on port 8501 (override with `PORT` or `STREAMLIT_SERVER_PORT`).

### Streamlit Community Cloud

The public app, `https://lisboa.streamlit.app/`, runs `app.py` from `main`.

| Item | Setting or behaviour |
|---|---|
| Python | 3.12, chosen in **Advanced settings** when deploying. Changing it later means deleting and redeploying the app, so note the subdomain and the secrets first |
| Dependencies | `requirements.txt` only: no torch and no local embedding model. Every package installs from a Linux wheel or a pure-Python source, so no `packages.txt` is needed |
| Secrets | The keys of [`.streamlit/secrets.example.toml`](../.streamlit/secrets.example.toml) in **App settings → Secrets**; root-level keys become environment variables. `config.py` reads them once, so reboot the app after editing them |
| Access | Apps from public repositories start public. Otherwise use **App settings → Sharing → This app is public and searchable**, or **Share → Make this app public**. Anonymous visitors see "You do not have access to this app or it does not exist" for a private app and for a subdomain no app uses |
| Hosted mode | `LISBOA_HOSTED="true"` in the secrets. The checkout path `/mount/src/<repo>` is also detected, but it is not a documented interface |
| Embeddings | With `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`, queries are embedded by Workers AI. Without them, semantic search is off: the startup log says so, place searches use the catalogue fallback, and the knowledge-base tool reports that it is unavailable |
| Data | The first session of a process downloads the vector DB from the `vector-db-latest` release and builds the Carris Urban and CP databases from the operators' GTFS feeds, all into `/tmp/lisboa_runtime`; the `transport-data-latest` release is used only when a feed fails. Every `LISBOA_DATA_REFRESH_HOURS` (6), a request starts a background check: the release is requested with its ETag, a newer `vector_db.zip` is installed next to the active copy and new searches switch to it, and the old copy is closed and deleted ten minutes later. The CP GTFS feed is re-checked at the same time; Carris Urban refreshes itself |
| Pushes | A push to `main` reaches the running app in almost real time: Streamlit re-imports the changed modules on the next rerun without restarting the process, so the downloaded data stay. A change to `requirements.txt` reinstalls and restarts the app. Reboot after a change to `config.py` |
| Sleep | Apps without traffic for 12 hours sleep; the next visit starts a cold process, data download included |
| Limits | About 690 MB to 2.7 GB of memory (February 2024 figures, subject to change). The local embedding model needs more, which is why hosted queries use Workers AI. With `requirements.txt`, a cold start (downloads and GTFS builds included) peaked near 300 MB in a local test |

Sources: Community Cloud documentation on [app dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), [Python version](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python), [secrets](https://docs.streamlit.io/develop/concepts/connections/secrets-management), [sharing](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app) and [app management](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

On **2026-12-01** the Hugging Face PRO subscription ends and the Space stops. Community Cloud is then the only hosted instance: disable `deploy_huggingface_space.yml` and its `HF_TOKEN` repository secret; the `Dockerfile`, `.dockerignore` and `scripts/hf_space_entrypoint.py` stay only for local Docker use.

### Hugging Face Spaces

Until 2026-12-01. The Docker image installs `requirements_local_embeddings.txt` with CPU-only torch and embeds queries locally.

`deploy_huggingface_space.yml` builds a deployment bundle with the Dockerfile, the application code, and a Space README, and uploads it to the Space `AndreSilvestre17/lisboa`:

- It requires an `HF_TOKEN` secret in the GitHub repository.
- The Space is created as a private Docker Space on the first deployment.
- The bundle includes the pricing metadata and the VisitLisboa and Lisboa Aberta source files, but not the vector database or the transport SQLite files. The Space downloads those from GitHub Releases at startup, into writable runtime storage.

### IPMA Relay

IPMA's network (FCT/RCTS, AS1930) does not accept TCP connections from AWS, where Hugging Face Spaces run: DNS resolves, but the connection times out before TLS. Other networks, including Cloudflare's, reach it normally. The Space therefore reads IPMA through a read-only relay on Cloudflare Pages, which forwards only `GET /open-data/...` requests to `https://api.ipma.pt` and caches each response for two minutes.

The relay runs on `pages.dev`, not `workers.dev`: the Space reaches `pages.dev`, but its connections to `workers.dev` time out. Community Cloud uses the relay too.

```js
// _worker.js (Cloudflare Pages, advanced mode): read-only relay for IPMA open data.
export default {
  async fetch(request) {
    const { pathname } = new URL(request.url);
    if (request.method !== "GET" || !pathname.startsWith("/open-data/")) {
      return new Response("Not found", { status: 404 });
    }
    return fetch(`https://api.ipma.pt${pathname}`, { cf: { cacheTtl: 120, cacheEverything: true } });
  },
};
```

1. Save the code above as `_worker.js` in an otherwise empty folder.
2. In Cloudflare, open Workers & Pages → Create application → Continue to Pages → Upload assets, name the project `ipma-relay`, and upload the folder.
3. Add the project URL (`https://ipma-relay.pages.dev`) as the `IPMA_BASE_URL` variable or secret of the hosted app and restart it.
4. The startup log must show `✅ IPMA weather layer: reachable (relay ...)`.

Local runs leave `IPMA_BASE_URL` unset and call IPMA directly.

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

- The store is always built with the local model, so these commands need `requirements_local_embeddings.txt`.
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
| Benchmark JSON outputs | `eval/results/Paper_2026-09/benchmark/` | `eval/run_benchmark.py` |
| Ablation JSON outputs | `eval/results/Paper_2026-09/ablation/` | `eval/run_ablation.py` |
| Statistical analysis (JSON and CSV) | `eval/results/Paper_2026-09/statistics/` | `eval/statistical_analysis.py` |
| Figures | `eval/results/Paper_2026-09/figures/` | The analysis notebook |

These are the paper evaluation (September 2026). The MSc thesis results (May 2026) are kept, unchanged, in `eval/results/MScThesis_2026-05/`; see the [Evaluation README](../eval/README.md#-versions).

The analysis notebook `eval/benchmark_ablation_analysis.ipynb` also exports the latest CSV summaries through `flatten_benchmark_results()` and `flatten_ablation_results()`:

- `eval/results/Paper_2026-09/benchmark/benchmark_flat_latest.csv`
- `eval/results/Paper_2026-09/benchmark/benchmark_summary_latest.csv`
- `eval/results/Paper_2026-09/ablation/ablation_flat_latest.csv`
- `eval/results/Paper_2026-09/ablation/ablation_summary_latest.csv`

## 🔄 GitHub Actions Automation

| Workflow | Trigger | Purpose | Main Output |
|---|---|---|---|
| `data_pipeline.yml` | Daily at **04:00 UTC**; manual runs with `events`, `places`, or `both` | Scrape VisitLisboa events daily and places weekly (7+ days after the last successful run, so a failure is retried the next day); each dataset is committed as soon as it is scraped | JSON artifacts committed under `data_collection/webscraping/` |
| `sync_vector_db.yml` | After each `Update Lisbon Data` run that committed new data, also when a later scrape failed; manual runs | Incremental ChromaDB sync in bounded batches | `vector_db.zip` (complete) or `vector_db_staging.zip` (in progress) on the `vector-db-latest` release |
| `sync_transport_runtime_data.yml` | Daily at **03:25 UTC**; pushes that change the transport release code; manual runs | Refresh the Carris Urban and CP runtime assets | Fixed-name ZIP files and a manifest on the `transport-data-latest` release |
| `deploy_huggingface_space.yml` | Pushes to `main` that change the app; completed vector or transport syncs; manual runs | Deploy the app (until 2026-12-01) | Updated Hugging Face Space |

Community Cloud needs no workflow: it follows `main` and picks up new release data through the background refresh.

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

### IPMA Weather Times Out on a Hosted Deployment

`IPMA ... failed after 3.00s: ConnectTimeout` with `⚠️ IPMA weather layer: unreachable (direct, ...)` at startup means the host cannot open a connection to IPMA. Configure the [IPMA relay](#ipma-relay). The answers then state that the weather data could not be obtained; they never claim that there are, or are not, active warnings.

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
