# 🧪 LISBOA Evaluation Pipeline

This folder contains the evaluation stack used for the LISBOA thesis workflow. It supports benchmark runs, ablation runs, response validators, statistical analysis artefacts, and analysis notebooks.

> [!IMPORTANT]
> The evaluation stack is **not** the same thing as the app quality gate. User-facing changes to agents, prompts, routing, QA, planners, or final response formatting must be validated with real LISBOA prompt runs through `scripts/run_prompts.py` and, where rendering matters, through Streamlit/browser inspection.

## 📍 What Lives Here

```text
eval/
|-- evaluation_groundtruth_queries.json
|-- evaluation_groundtruth_queries_demo.json
|-- evaluation_groundtruth_queries_paper_eval.json
|-- paper_eval_annotations.json
|-- run_benchmark.py
|-- run_ablation.py
|-- runtime_utils.py
|-- llm_judge.py
|-- constraint_judge.py
|-- merge_ablation.py
|-- statistical_analysis.py
|-- paper_eval_analysis.py
|-- validators/
|   |-- response_heuristics.py
|-- tests/
|   |-- README.md
|   |-- test_dataset_integrity.py
`-- results/
```

`eval/tests/` is intentionally lean. It protects deterministic integrity only:
dataset shape, tool references, and validator helper behaviour. Do not restore
large mocked judge suites or strict prompt-coverage manifests as the default
quality gate.

## 🚦 Evaluation Modes

| Mode | Entrypoint | Purpose |
|---|---|---|
| Lean deterministic checks | `python -m pytest eval/tests/ -q` | Dataset and validator integrity |
| Benchmark | `python -m eval.run_benchmark --dataset eval/evaluation_groundtruth_queries.json` | Isolated worker-agent evaluation |
| Ablation | `python -m eval.run_ablation --dataset eval/evaluation_groundtruth_queries.json` | Zero-shot vs LISBOA comparison |
| Prompt smoke | `python scripts/run_prompts.py --suite smoke` | Real LISBOA execution path |

Benchmark and ablation runners require module form (`python -m eval.run_benchmark`
and `python -m eval.run_ablation`) so repository imports resolve correctly.

## 🧪 Shared Evaluation Corpus

The primary corpus is `evaluation_groundtruth_queries.json`. It currently
contains 72 entries across 6 domains:

| Domain | Count |
|---|---:|
| `weather` | 13 |
| `transport` | 36 |
| `researcher` | 13 |
| `multi_agent` | 3 |
| `greeting` | 3 |
| `out_of_scope` | 4 |

The corpus is for realistic evaluation scenarios, not exhaustive exported-tool coverage.

> [!TIP]
> Tool counts can change. Verify [`tools/__init__.py`](../tools/__init__.py).

## 📝 Paper Evaluation (RINENG Revision)

The revision adds ten itinerary requests (`M04` to `M13`) to the ablation corpus and
reruns the whole ablation on the revised system under one protocol: the same two model
profiles, the same judges and judge prompt as the May 2026 run, and a fresh LISBOA
session for every query. The May artefacts stay in `results/` as the record of the
submitted version; they are not pooled with the rerun, because the May run shared one
conversation across queries and evaluated older code.

| File | Content |
|---|---|
| `evaluation_groundtruth_queries_paper_eval.json` | The 72 original queries, unchanged, plus the ten itinerary requests |
| `paper_eval_annotations.json` | Expected agents for every ablation query; explicit constraints for every itinerary request |
| `constraint_judge.py` | Checklist judge: marks each itinerary constraint as met, not met, or not assessable, blind to the condition |
| `paper_eval_analysis.py` | Provenance, quality tests, latency and cost, routing and QA paths, judge reliability, and constraints, as JSON and Markdown |
| `merge_ablation.py` | Joins runs of the same protocol and code that cover different queries (for example, a run split by query); refuses mixed protocols unless `--allow-mixed-protocol` is given for an archival merge |

Run order (the ablation takes about five to six hours for both profiles; split it by profile if needed):

```powershell
python -X utf8 -m pytest eval/tests/ -q
$env:VECTOR_DB_RELEASE_FORCE_DOWNLOAD = 'true'
python -X utf8 -u -m eval.run_ablation --dataset eval/evaluation_groundtruth_queries_paper_eval.json --fresh-session --only-profile closed_source --output-prefix ablation_final
python -X utf8 -u -m eval.run_ablation --dataset eval/evaluation_groundtruth_queries_paper_eval.json --fresh-session --only-profile open_source --resume eval/results/ablation/ablation_final_<timestamp>.partial.jsonl
python -X utf8 -u -m eval.run_benchmark --dataset eval/evaluation_groundtruth_queries_paper_eval.json --output-prefix benchmark_final
python -X utf8 -m eval.constraint_judge --ablation eval/results/ablation/ablation_final_<timestamp>.json
python -X utf8 -m eval.statistical_analysis --ablation eval/results/ablation/ablation_final_<timestamp>.json --benchmark eval/results/benchmark/benchmark_final_<timestamp>.json --output-prefix statistical_analysis_final
python -X utf8 -m eval.paper_eval_analysis --ablation eval/results/ablation/ablation_final_<timestamp>.json --benchmark eval/results/benchmark/benchmark_final_<timestamp>.json --constraints eval/results/constraints/constraint_checklist_<timestamp>.json
```

The benchmark and the ablation run on the same commit, so every reported number comes from one version of the system. The benchmark runs the 62 worker queries of the same corpus (multi-agent, greeting, and out-of-scope rows have no isolated worker). The newest `ablation_final_<timestamp>.json` and `benchmark_final_<timestamp>.json` are the ones the notebook reads. `paper_eval_analysis` reports latency and cost for both: per model inside the workers (benchmark) and per model with and without LISBOA (ablation). Keep only the final run in `eval/results/`: remove interim and superseded result files before committing the new ones.

- `--fresh-session` resets LISBOA's conversation state before every query. Without it, the runner behaves as in May 2026, when one conversation carried over from query to query.
- `--query-id` runs only the listed queries, in corpus order.
- Each finished comparison is appended to `<prefix>_<timestamp>.partial.jsonl`. If a run stops, repeat the command with `--resume <checkpoint>`; add `--retry-errors` to rerun comparisons where a response or a judge call failed. A resume is refused when the query selection, the session protocol, the judges, or the system code differ from the checkpoint.
- The runner sets the Azure deployment to the model of each profile and refuses to start a profile whose agents use another model.
- Every artefact records its provenance: commit, branch, every uncommitted or new code and evaluation file, a SHA-256 over the system code as it was on disk, the last commit that touched the system code, package versions, the model version reported by the API, input-file hashes, the tool registry, and the local data snapshots.

> [!IMPORTANT]
> Commit the code, corpus, and annotations before the run, so the provenance points to a clean commit. The runtime never refreshes the local vector database; set `VECTOR_DB_RELEASE_FORCE_DOWNLOAD=true` for the first start on the day of the run so events and places come from the latest release. Run in daytime: "now" transport questions asked at night receive closed-service answers.

## ☑️ Recommended Validation

Use this sequence after code changes:

```powershell
python -X utf8 scripts/syntax_check.py
python -X utf8 -m pytest eval/tests/ -q
python -X utf8 scripts/run_prompts.py --suite smoke
```

For a focused agent or prompt change, run at least one direct prompt plus one variant with a different entity, language, location, or wording. For transport logic, also consider:

```powershell
python -X utf8 scripts/run_transport_verification.py
```

> [!NOTE]
> The legacy `tests/` directory and its strict live-coverage suite were retired during the 2026-05 cleanup. Live integrations are now exercised through real prompt smoke runs and the operator-specific verification script.

## 📂 Outputs

Evaluation artefacts are written under `eval/results/`, usually in one of these
subfolders:

- `benchmark/`
- `ablation/`
- `constraints/`
- `statistics/`
- `figures/`

Keep result interpretation explicit: separate benchmark, ablation, deterministic
validator checks, live/prompt smoke runs, and user-study evidence.
