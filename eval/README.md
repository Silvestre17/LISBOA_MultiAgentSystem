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
|   |-- test_paper_eval_pipeline.py
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
reruns the benchmark and the whole ablation on the revised system under one protocol:
the same two model profiles, the same judges and judge prompt as the May 2026 run, and a
fresh LISBOA session for every query. Only the final run is kept in `results/`; the May
and interim artefacts remain in the Git history.

| File | Content |
|---|---|
| `evaluation_groundtruth_queries_paper_eval.json` | The 72 original queries, unchanged, plus the ten itinerary requests |
| `paper_eval_annotations.json` | Expected agents for every ablation query; explicit constraints for every itinerary request |
| `constraint_judge.py` | Checklist judge: marks each itinerary constraint as met, not met, or not assessable, blind to the condition |
| `paper_eval_analysis.py` | Provenance and protocol checks, quality tests, latency and cost, routing and QA paths, judge reliability, and constraints, as JSON and Markdown |
| `benchmark_ablation_analysis.ipynb` | Figures 3 and 4, Tables 5 to 7, a value sheet with every number of the manuscript and the response letter, and an extended analysis of both runs per model and for both models together (distributions, every breakdown, per-query gains, latency, cost, tokens, pipeline paths, constraints, judges) |
| `merge_ablation.py` | Joins runs of the same protocol and code that cover different queries (for example, a run split by query); refuses mixed protocols unless `--allow-mixed-protocol` is given for an archival merge |

Run order, in PowerShell from the repository root. The ablation takes about five to six
hours for both profiles and the benchmark about one; run in daytime, because "now"
transport questions asked at night receive closed-service answers.

```powershell
python -X utf8 -m pytest eval/tests/ -q

# 1. Download the latest vector database once, then keep it fixed for the whole run.
$env:VECTOR_DB_RELEASE_FORCE_DOWNLOAD = 'true'
python -X utf8 -c "from agent.utils.vector_db_release import ensure_vector_db_from_release as ensure; print(ensure().message)"
$env:VECTOR_DB_RELEASE_FORCE_DOWNLOAD = 'false'

# 2. Ablation, one profile per session; the second call completes the same file.
python -X utf8 -u -m eval.run_ablation --dataset eval/evaluation_groundtruth_queries_paper_eval.json --fresh-session --only-profile closed_source --output-prefix ablation_final
python -X utf8 -u -m eval.run_ablation --dataset eval/evaluation_groundtruth_queries_paper_eval.json --fresh-session --only-profile open_source --resume eval/results/ablation/ablation_final_<timestamp>.partial.jsonl

# 3. Worker benchmark, both response models.
python -X utf8 -u -m eval.run_benchmark --dataset eval/evaluation_groundtruth_queries_paper_eval.json --output-prefix benchmark_final

# 4. Constraint checklist, statistics, and the paper analysis on the final files.
python -X utf8 -m eval.constraint_judge --ablation eval/results/ablation/ablation_final_<timestamp>.json
python -X utf8 -m eval.statistical_analysis --ablation eval/results/ablation/ablation_final_<timestamp>.json --benchmark eval/results/benchmark/benchmark_final_<timestamp>.json --output-prefix statistical_analysis_final
python -X utf8 -m eval.paper_eval_analysis --ablation eval/results/ablation/ablation_final_<timestamp>.json --benchmark eval/results/benchmark/benchmark_final_<timestamp>.json --constraints eval/results/constraints/constraint_checklist_<timestamp>.json

# 5. Notebook: figures, tables, and the value sheet.
python -X utf8 -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=lisboa_thesis2026 eval/benchmark_ablation_analysis.ipynb
```

The `<timestamp>` of the ablation JSON is the one printed by the second ablation call:
that file holds both profiles. The first call also writes a JSON with one profile; it is
superseded and can be deleted. The benchmark runs the 62 worker queries of the same corpus
(multi-agent, greeting, and out-of-scope rows have no isolated worker).

- `--fresh-session` resets LISBOA's conversation state before every query. Without it, the runner behaves as in May 2026, when one conversation carried over from query to query.
- `--query-id` runs only the listed queries, in corpus order, in both runners.
- Both runners append each finished answer to `<prefix>_<timestamp>.partial.jsonl`. If a run stops, repeat the command with `--resume <checkpoint>` (the file keeps the checkpoint's prefix); add `--retry-errors` to rerun answers where a response or a judge call failed. A resume is refused when the query selection, the judges, the system code, or (ablation) the session protocol differ from the checkpoint.
- The console shows, for every answer, the score of each judge, the latency, the response cost, and, for LISBOA, the tools, the agents called, the execution type, and the QA path, followed by progress and the estimated time left; each run ends with a summary per model and warns when the file is incomplete.
- The ablation runner sets the Azure deployment to the model of each profile and refuses to start a profile whose agents use another model. The benchmark records the model name the API reports for every worker call and flags any call to another model.
- Every artefact records its provenance: commit, branch, every uncommitted or new code and evaluation file, a SHA-256 over the system code as it was on disk, the last commit that touched the system code, package versions, the model version reported by the API, input-file hashes, the tool registry, and the local data snapshots at the start and end of the run.
- The notebook and `paper_eval_analysis` check that both runs are complete, used one system-code fingerprint and one corpus file, had no response or judge errors, and that the constraint checklist belongs to the loaded ablation file. The notebook saves every manuscript value in `results/statistics/paper_values_<ablation file>.csv` and the tidy per-answer tables in `results/statistics/extended_tables_<ablation file>.xlsx`.

To rehearse the whole pipeline on a few queries without touching `results/`, point
`LISBOA_EVAL_RESULTS_DIR` at another folder; the runners, the analyses, and the notebook
all read and write there:

```powershell
$env:LISBOA_EVAL_RESULTS_DIR = "$env:TEMP\lisboa_rehearsal"
python -X utf8 -u -m eval.run_ablation --dataset eval/evaluation_groundtruth_queries_paper_eval.json --fresh-session --query-id W01,T02,M04 --output-prefix ablation_final
Remove-Item Env:LISBOA_EVAL_RESULTS_DIR
```

> [!IMPORTANT]
> Commit the code, corpus, and annotations before the run, so the provenance points to a clean commit, and do not pull or edit code until the run ends. Keep only the final run in `eval/results/`: remove interim and superseded result files before committing the new ones.

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
