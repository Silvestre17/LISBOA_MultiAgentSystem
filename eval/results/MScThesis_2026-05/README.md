# 🎓 MSc Thesis Evaluation (May 2026)

This folder holds the evaluation results reported in the MSc thesis *LISBOA: A Multi-Agent Approach for Personalized Tourism and Urban Mobility in Lisbon* (NOVA IMS, 2025/2026). The files, including the analysis notebook with its outputs, are unchanged from the thesis version of the repository and keep their original names.

The results of the paper evaluation (September 2026) are in [`../Paper_2026-09/`](../Paper_2026-09/). Both runs use the same judges, judge prompt, and model profiles; the [Evaluation README](../../README.md#-versions) lists what differs between them.

| Item | Thesis run |
|---|---|
| Corpus | [`evaluation_groundtruth_queries.json`](../../evaluation_groundtruth_queries.json), 72 queries |
| Worker benchmark | 62 queries, 124 scored responses, run on 2026-05-15 |
| Paired ablation | 65 queries, zero-shot vs LISBOA, run on 2026-05-15 |
| Response models | GPT-5.4-mini and Kimi-K2.5 |
| Judges | GPT-5.4-mini and Kimi-K2.5, scores averaged |

## 📂 Contents

| Folder | Files |
|---|---|
| `benchmark/` | `benchmark_final_20260515_101309.json` |
| `ablation/` | `ablation_final_20260515_154508.json` |
| `statistics/` | `statistical_analysis_final_20260515_154656.json` and its benchmark and ablation CSV tables; `judge_agreement_validation_20260523.csv`; `paper_results_claim_validation_20260523.csv` |
| `figures/` | `benchmark_quality` and `ablation_quality_lift`, as PNG and SVG |
| `benchmark_ablation_analysis.ipynb` | The analysis notebook of the thesis, with the outputs it produced from these files |

> [!NOTE]
> The runners, the analyses, and the notebook in `eval/` read and write in `../Paper_2026-09/`, so they do not change these files. This folder is a record of the thesis results: the notebook here is kept for its saved outputs and is not meant to be run from this folder.
