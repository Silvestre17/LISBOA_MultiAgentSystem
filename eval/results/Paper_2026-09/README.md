# 📝 Paper Evaluation (September 2026)

This folder holds the evaluation results reported in the journal article on LISBOA, revised for *Results in Engineering*. The runners, the analyses, and the notebook read and write here by default.

The results of the MSc thesis (May 2026) are in [`../MScThesis_2026-05/`](../MScThesis_2026-05/). Both runs use the same judges, judge prompt, and model profiles; the [Evaluation README](../../README.md#-versions) lists what differs between them.

| Item | Paper run |
|---|---|
| Corpus | [`evaluation_groundtruth_queries_paper_eval.json`](../../evaluation_groundtruth_queries_paper_eval.json), 82 queries: the 72 of the thesis corpus, unchanged, plus ten itinerary requests (`M04` to `M13`) |
| Worker benchmark | 62 queries, 124 scored responses, run on 2026-09-26 |
| Paired ablation | 75 queries, zero-shot vs LISBOA, a fresh LISBOA session for every query, run on 2026-09-26 |
| Response models | GPT-5.4-mini and Kimi-K2.5 |
| Judges | GPT-5.4-mini and Kimi-K2.5, scores averaged |
| Provenance | Recorded inside each JSON: commit, uncommitted files, and a SHA-256 of the system code at run time |

## 📂 Contents

| Folder | Files |
|---|---|
| `benchmark/` | `benchmark_final_20260926_145914.json` |
| `ablation/` | `ablation_final_20260926_141018.json` |
| `constraints/` | `constraint_checklist_20260926_212640.json`, the constraint checklist of the itinerary requests |
| `statistics/` | `statistical_analysis_final_20260926_150320` (JSON and CSV tables); `paper_eval_analysis_20260926_213046` (JSON and Markdown); `paper_values_ablation_final_20260926_141018.csv`, every value of the manuscript; `extended_tables_ablation_final_20260926_141018.xlsx`, the tidy per-answer tables |
| `figures/` | `benchmark_quality` and `ablation_quality_lift` (Figures 3 and 4 of the article), as PDF, PNG, and SVG |

The commands that produce these files are in the [Evaluation README](../../README.md#-paper-evaluation-rineng-revision).
