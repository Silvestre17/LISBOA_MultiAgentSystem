# 📂 Evaluation Results

LISBOA was evaluated twice with the same judges, judge prompt, and model profiles. Each evaluation has its own folder.

| Folder | Evaluation | Corpus | Benchmark | Ablation |
|---|---|---|---|---|
| [`MScThesis_2026-05/`](./MScThesis_2026-05/) | MSc thesis, May 2026 | 72 queries | 62 queries, 124 responses | 65 queries |
| [`Paper_2026-09/`](./Paper_2026-09/) | Journal article (*Results in Engineering*, revision), September 2026 | 82 queries (the 72, plus ten itinerary requests) | 62 queries, 124 responses | 75 queries |

Each folder has its own README with the files it contains. The [Evaluation README](../README.md#-versions) explains what differs between the two runs. New runs write to `Paper_2026-09/` unless `LISBOA_EVAL_RESULTS_DIR` points elsewhere.
