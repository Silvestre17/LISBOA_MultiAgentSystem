# 📊 LISBOA Evaluation Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Ground_Truth-72_Queries-0A7E07?style=for-the-badge" alt="72 queries">
  <img src="https://img.shields.io/badge/Exported_Tools-45-0A7E07?style=for-the-badge" alt="45 tools">
  <img src="https://img.shields.io/badge/Benchmark-Isolated_Workers-6A1B9A?style=for-the-badge" alt="Benchmark isolated workers">
  <img src="https://img.shields.io/badge/Ablation-Zero--shot_vs_LISBOA-C77800?style=for-the-badge" alt="Ablation">
  <img src="https://img.shields.io/badge/Coverage-Strict_Live_Suite-0052CC?style=for-the-badge" alt="Strict live coverage">
</p>

<p align="center">
  <strong>Research-grade evaluation for benchmark, ablation, strict live tool coverage, calibration, and analysis artefacts in the LISBOA multi-agent thesis project.</strong>
</p>

## 📍 Overview

This folder contains the evaluation stack that supports the thesis claims around grounded tool use, response quality, and operational reproducibility.

The pipeline combines:

- LLM-as-a-Judge scoring
- deterministic tool metrics
- deterministic response heuristics
- deterministic Metro route validation
- strict live coverage of the exported tool registry
- calibration support for human vs judge comparison

> Repository overview: [`../README.md`](../README.md)

## 🔗 Quick Links

- [🧱 Evaluation workspace layout](#-evaluation-workspace-layout)
- [🧪 Shared evaluation corpus](#-shared-evaluation-corpus)
- [📏 What the pipeline measures](#-what-the-pipeline-measures)
- [🚦 Execution modes](#-execution-modes)
- [📦 Outputs and artefacts](#-outputs-and-artefacts)
- [▶️ Useful commands](#-useful-commands)
- [🔐 Environment requirements](#-environment-requirements)
- [📝 Interpretation notes](#-interpretation-notes)

## 📊 At a glance

| Item | Current value |
|------|---------------|
| Ground-truth dataset | `evaluation_groundtruth_queries.json` |
| Evaluation entries | 72 |
| Dataset domains | 6 |
| Exported runtime tools covered by manifest | 45 |
| Benchmark runner | `run_benchmark.py` |
| Ablation runner | `run_ablation.py` |
| Judge | `llm_judge.py` |
| Deterministic validators | `validators/` |
| Analysis notebook | `benchmark_ablation_analysis.ipynb` |

## 🧱 Evaluation workspace layout

```text
eval/
├── evaluation_groundtruth_queries.json  # Shared corpus; ablation filters to grounded domains by default
├── llm_judge.py                         # LLM-as-a-Judge with structured scoring
├── run_benchmark.py                     # Isolated worker benchmark runner
├── run_ablation.py                      # Zero-shot vs LISBOA comparison runner
├── runtime_utils.py                     # Fingerprints, metadata, cost helpers
├── benchmark_ablation_analysis.ipynb    # Analysis notebook and latest CSV exports
├── human_calibration/
│   ├── calibration_template.json
│   └── run_calibration.py               # Human vs judge agreement analysis
├── validators/
│   ├── response_heuristics.py           # LLM-free response checks
│   └── transport_validator.py           # Deterministic Metro validation
├── tests/
│   ├── test_benchmark_utils.py
│   ├── test_cost_accounting.py
│   ├── test_dataset_integrity.py
│   ├── test_llm_judge.py
│   └── test_validators.py
└── results/
    ├── benchmark/
    ├── ablation/
    └── figures/
```

Related repository-level coverage assets live under `tests/`:

```text
tests/
├── conftest.py                      # Strict live prerequisite enforcement
├── fixtures/
│   └── tool_coverage_manifest.json  # 45-tool prompt coverage manifest
└── test_tool_prompt_coverage.py     # Strict live worker-agent coverage suite
```

Manual repository runners now live under `scripts/`:

```text
scripts/
├── run_prompts.py                   # Manual smoke and coverage runner
├── run_transport_verification.py    # Transport verification harness
└── syntax_check.py                  # Syntax smoke checker
```

The fast manifest-integrity checks now live in `eval/tests/test_dataset_integrity.py`.

## 🧪 Shared evaluation corpus

The shared dataset lives in `evaluation_groundtruth_queries.json` and currently contains **72 entries across 6 domains**. Exhaustive exported-tool coverage is enforced separately by the strict live manifest in `tests/fixtures/tool_coverage_manifest.json`, so the main corpus can stay focused on realistic user-facing evaluation.

For demonstrations, the repository also includes `evaluation_groundtruth_queries_demo.json`, a tiny two-query walkthrough corpus intended to show the mechanics of the runners without launching a full evaluation cycle.

| Domain | Count | Coverage |
|--------|------:|----------|
| `weather` | 13 | IPMA forecasts, warnings, current weather, Portugal-wide overview |
| `transport` | 36 | Metro, Carris Metropolitana, Carris Urban, CP, multimodal routing |
| `researcher` | 13 | places, events, open data, semantic retrieval, web fallback |
| `multi_agent` | 3 | cross-domain itinerary and planning queries |
| `greeting` | 3 | greeting and lightweight assistant behavior |
| `out_of_scope` | 4 | unsupported or off-scope requests |

The dataset is designed so that `expected_tools` collectively reference the exported tool registry, and `eval/tests/test_dataset_integrity.py` checks that integrity.

### Entry schema

```json
{
  "id": "T02",
  "query": "How do I get from Baixa-Chiado to Aeroporto?",
  "domain": "transport",
  "expected_tools": ["get_route_between_stations"],
  "expected_facts": ["Start at Baixa-Chiado", "Transfer at Alameda"],
  "language": "en",
  "edge_case": false,
  "edge_type": null,
  "expected_behavior": null
}
```

## 📏 What the pipeline measures

### 🤖 LLM-as-a-Judge

`llm_judge.py` uses configurable Azure-compatible judges with structured output to score five dimensions on a 1 to 5 scale:

- factual accuracy
- tool usage
- completeness
- relevance
- response quality

As of 2026-04, the benchmark and ablation runners can evaluate the same response with more than one judge model, persist every raw `judge_runs` entry, and store an averaged compatibility `scores` block so downstream readers do not need to recompute the mean manually.

If one judge fails, that failed run is still persisted in `judge_runs` and flagged in `judge_summary`, but it is excluded from the averaged compatibility `scores` block.

The default evaluation matrix is now a closed plus open judge pair:

- `azure::gpt-5.4-mini`
- `azure::Kimi-K2.5`

This matrix can be overridden with repeatable `--judge-model-spec provider::model` flags or with `EVAL_JUDGE_MODEL_SPECS`.

Bias controls already present in the judge flow include:

- reasoning-first scoring
- explicit rubric descriptors
- anti-mean-reversion guidance
- expected vs actual tool comparison in the prompt
- `temperature = 0` for deterministic judgment

### 🧮 Deterministic metrics and validators

The evaluation pipeline also computes non-LLM checks:

| Component | File | Purpose |
|-----------|------|---------|
| Tool Precision, Recall, F1 | `run_benchmark.py` and `run_ablation.py` | compare observed tools against `expected_tools` |
| Response heuristics | `validators/response_heuristics.py` | detect tool leaks, bad length, language mismatch, unsupported capability claims, heavy emoji use |
| Metro route validator | `validators/transport_validator.py` | validate station existence, line membership, transfer correctness, and route facts |
| Error categorization | `runtime_utils.py` | normalize runtime failures into stable evaluation categories |

### 🧾 Reproducibility, token, and cost metadata

Persisted outputs can include:

- ground-truth dataset fingerprint
- tool registry fingerprint
- response model metadata and configs
- evaluation model metadata and configs
- multi-judge metadata including `evaluation_models`, `judge_model_configs`, and per-record `judge_runs`
- token usage blocks for response, evaluation, and combined calls
- optional cost accounting via `pricing_by_model`
- per-call usage breakdown for the LISBOA arm when available
- per-agent usage and per-agent cost blocks for LISBOA ablation records when available

Expected pricing structure:

```python
pricing_by_model = {
    "pricing_source": "https://www.llm-prices.com/current-v1.json",
  "pricing_updated_at": "2026-04-17",
    "models": {
    "provider::model": {"input": 0.25, "output": 2.0},
    },
}
```

Flat mappings also work.

## 🚦 Execution modes

| Mode | Main entrypoint | What it evaluates | Uses live services | Main output |
|------|-----------------|-------------------|--------------------|-------------|
| Fast deterministic validation | `eval/tests/` | utilities, dataset integrity, judge helpers, validators | no | test results only |
| Benchmark | `eval/run_benchmark.py` | isolated worker agents for `weather`, `transport`, `researcher` | yes | `eval/results/benchmark/benchmark_results_<timestamp>.json` |
| Ablation | `eval/run_ablation.py` | zero-shot vs LISBOA full pipeline | yes | `eval/results/ablation/ablation_results_<timestamp>.json` |
| Strict live coverage | `tests/test_tool_prompt_coverage.py` | live usage of the exported tool registry | yes | `eval/results/coverage/coverage_results_<timestamp>.json` |
| Calibration | `eval/human_calibration/run_calibration.py` | human vs judge agreement | no, uses saved artefacts | `eval/results/calibration/calibration_summary_<timestamp>.json` |
| Notebook analysis | `benchmark_ablation_analysis.ipynb` | flattening, summaries, figures | no, uses saved artefacts | latest CSVs and optional figure exports |

### Benchmark scope

`run_benchmark.py` evaluates isolated worker behavior, not the full multi-agent assistant. It currently targets:

- `weather`
- `transport`
- `researcher`

Each output record can contain averaged compatibility `scores`, raw `judge_runs`, `scores_by_judge`, tool metrics, heuristics, latency, SLA compliance, model metadata, and organized response/evaluation/combined cost blocks.

### Ablation scope

`run_ablation.py` compares two fair profile pairs:

- a **closed-source pair**, where zero-shot and LISBOA both use the closed response profile
- an **open-model pair**, where zero-shot and LISBOA both use the open response profile

Within each pair, the zero-shot and LISBOA arms are judged by the same multi-judge matrix. By default, ablation runs focus on `weather`, `transport`, `researcher`, and `multi_agent` queries, excluding `greeting` and `out_of_scope` shortcuts because LISBOA handles those through supervisor-level direct responses rather than the grounded pipeline under study. Persisted ablation records keep the primary compatibility `metrics` block for the primary profile and store every profile-specific comparison under `comparisons`.

### Strict live coverage scope

The strict live suite is intentionally separate from quick tests because it:

- requires real credentials
- hits real services
- depends on live LLM routing behavior
- is intended for evaluation sign-off, not every local iteration

## 📦 Outputs and artefacts

### JSON artefacts

| Artefact | Default path | Notes |
|----------|--------------|-------|
| Benchmark results | `eval/results/benchmark/benchmark_results_<timestamp>.json` | isolated worker evaluation |
| Ablation results | `eval/results/ablation/ablation_results_<timestamp>.json` | zero-shot vs LISBOA comparison |
| Coverage results | `eval/results/coverage/coverage_results_<timestamp>.json` | created when the strict live suite runs |
| Calibration summary | `eval/results/calibration/calibration_summary_<timestamp>.json` | created when calibration runs |

### Notebook exports

`benchmark_ablation_analysis.ipynb` also writes latest CSV summaries through its flattening helpers:

- `eval/results/benchmark/benchmark_flat_latest.csv`
- `eval/results/benchmark/benchmark_summary_latest.csv`
- `eval/results/ablation/ablation_flat_latest.csv`
- `eval/results/ablation/ablation_summary_latest.csv`

When plots are exported, figure artefacts can also appear under `eval/results/figures/`, for example the ablation SVG summaries already present in the repository.

### What the live coverage artefact stores

For each coverage-manifest prompt, the persisted coverage JSON can store:

- expected tools
- observed tools
- tool precision, recall, and F1
- latency
- runtime errors and normalized error categories
- retrieved context
- response-model metadata by worker domain
- validated prerequisite names only, never secret values

## ▶️ Useful commands

### Fast validation before slower runs

```bash
python -m pytest eval/tests/test_dataset_integrity.py eval/tests/test_benchmark_utils.py eval/tests/test_cost_accounting.py eval/tests/test_validators.py eval/tests/test_llm_judge.py -v
```

### Benchmark and ablation

```bash
python eval/run_benchmark.py --mode run_test
python eval/run_benchmark.py --mode full
python eval/run_benchmark.py --limit 5
python eval/run_benchmark.py --limit 3 --judge-model-spec azure::gpt-5.4-mini --judge-model-spec azure::Kimi-K2.5
python eval/run_ablation.py --mode run_test
python eval/run_ablation.py --mode full
python eval/run_ablation.py --limit 3 --open-model-spec azure::Kimi-K2.5 --judge-model-spec azure::gpt-5.4-mini --judge-model-spec azure::Kimi-K2.5
```

Both runners now auto-load the checked-in pricing catalog from `data/pricing/llm_model_pricing.json` when no explicit `pricing_by_model` payload is injected programmatically, so CLI-generated artefacts include cost accounting by default.

### Tiny demo dataset walkthrough

```bash
python eval/run_ablation.py --dataset eval/evaluation_groundtruth_queries_demo.json --open-model-spec azure::Kimi-K2.5 --output-prefix ablation_demo
python eval/run_benchmark.py --dataset eval/evaluation_groundtruth_queries_demo.json --limit 1 --output-prefix benchmark_demo
```

Use the ablation command above to demonstrate the full zero-shot versus LISBOA comparison flow on a tiny corpus. The benchmark command is intentionally limited because the demo JSON includes a `multi_agent` entry, while the isolated benchmark only evaluates the worker-agent domains. The custom output prefixes keep demo artefacts persisted without replacing the main `benchmark_results_*` and `ablation_results_*` files that power the analysis notebook.

### Coverage and prompt-driven checks

```bash
python scripts/run_prompts.py --suite smoke
python scripts/run_prompts.py --suite coverage
python scripts/run_prompts.py --suite coverage --limit 5
python -m pytest tests/test_tool_prompt_coverage.py --run-live -m "live and coverage" -v
```

### Calibration

```bash
python eval/human_calibration/run_calibration.py --human eval/human_calibration/calibration_filled.json --benchmark eval/results/benchmark/benchmark_results_YYYYMMDD_HHMMSS.json
python eval/human_calibration/run_calibration.py --human eval/human_calibration/calibration_filled.json --benchmark eval/results/benchmark/benchmark_results_YYYYMMDD_HHMMSS.json --judge-source azure::gpt-5.4-mini
```

Use `--judge-source average` (default) to read the compatibility-average `scores` block, or pass a specific judge model id from `judge_runs` to calibrate one judge independently.

## 🔐 Environment requirements

### Benchmark and ablation

Credentials depend on the active provider configured in `config.py`. The current documented default provider is Azure, so the usual minimum is:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`

### Strict live coverage

The strict live suite is designed to fail loudly if required prerequisites are missing or still templated, including:

- active provider credentials
- `METRO_CONSUMER_KEY`
- `METRO_CONSUMER_SECRET`
- `TAVILY_API_KEY`
- local runtime assets such as the vector DB and transport support databases

This behavior is deliberate. For evaluation sign-off, hidden skips are worse than explicit failures.

## 📝 Interpretation notes

- Use deterministic validation first, then live coverage, then benchmark or ablation.
- Judge scores complement deterministic checks. They do not replace them.
- The authoritative runtime tool count comes from `tools/__init__.py`.
- `eval/results/coverage/` and `eval/results/calibration/` may not exist until their respective workflows are run.
- Keeping smoke, coverage, benchmark, and ablation layers separate makes regressions easier to localize.

## 📚 References

- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). *Judging LLM-as-a-judge with MT-bench and Chatbot Arena.* Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, 46595-46623.
- Lee, C., Zeng, T., Jeong, J., Sohn, J., & Lee, K. (2026). *How to Correctly Report LLM-as-a-Judge Evaluations* (arXiv:2511.21140). arXiv. https://doi.org/10.48550/arXiv.2511.21140
