# LISBOA Evaluation Pipeline

This folder contains the evaluation stack used for the LISBOA thesis workflow.
It supports benchmark runs, ablation runs, response validators, statistical
analysis artefacts, and analysis notebooks.

The evaluation stack is not the same thing as the app quality gate. User-facing
changes to agents, prompts, routing, QA, planners, or final response formatting
must be validated with real LISBOA prompt runs through `scripts/run_prompts.py`
and, where rendering matters, through Streamlit/browser inspection.

## 📍 What Lives Here

```text
eval/
|-- evaluation_groundtruth_queries.json
|-- evaluation_groundtruth_queries_demo.json
|-- run_benchmark.py
|-- run_ablation.py
|-- runtime_utils.py
|-- llm_judge.py
|-- validators/
|   |-- response_heuristics.py
|   `-- transport_validator.py
|-- tests/
|   |-- README.md
|   |-- test_dataset_integrity.py
|   `-- test_validators.py
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

The corpus is for realistic evaluation scenarios, not exhaustive exported-tool
coverage. Tool counts can change; verify `tools/__init__.py` before making exact
claims in thesis or documentation text.

## ☑️ Recommended Validation

Use this sequence after code changes:

```powershell
python -X utf8 scripts/syntax_check.py
python -X utf8 -m pytest tests/ eval/tests/ -q
python -X utf8 scripts/run_prompts.py --suite smoke
```

For a focused agent or prompt change, run at least one direct prompt plus one
variant with a different entity, language, location, or wording. For transport
logic, also consider:

```powershell
python -X utf8 scripts/run_transport_verification.py
python -X utf8 -m pytest tests/test_lisbon_transport.py -q --run-live -m live
```

The live pytest command is opt-in because it calls external providers.

## 📂 Outputs

Evaluation artefacts are written under `eval/results/`, usually in one of these
subfolders:

- `benchmark/`
- `ablation/`
- `statistics/`
- `figures/`

Keep result interpretation explicit: separate benchmark, ablation, deterministic
validator checks, live/prompt smoke runs, and user-study evidence.
