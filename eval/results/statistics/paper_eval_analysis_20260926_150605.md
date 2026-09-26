# Paper evaluation analysis

Generated 2026-09-26T15:05:31 from `eval/results/ablation/ablation_final_20260926_141018.json`.

## Provenance

- Commit `2a84c3933da96d639a2aa57abbf2029341884cd1` on `lisboa-paper-eval`; uncommitted changes: True
- Last system-code commit: 2026-09-25T22:02:04+01:00 (LISBOA - 25/09/2026 (General fixes and improvements))
- Run: 2026-09-26T10:55:45.937928 to 2026-09-26T14:10:18.369642; sessions: 2
- Queries: 75; tools: 45; API model names: {'closed_source': ['gpt-5.4-mini-2026-03-17'], 'open_source': ['Kimi-K2.5']}
- Fresh session: 150 of 150 LISBOA responses; execution summaries: 150
- Checks: single_commit OK, clean_working_tree FAILED, fresh_session_where_requested OK, no_response_errors OK, no_judge_errors OK, lisboa_calls_use_profile_model OK
- Response errors: none; judge errors: none
- Benchmark: 124 of 124 responses; run 2026-09-26T14:10:32.548891 to 2026-09-26T14:59:14.294090; API model names: {'azure::gpt-5.4-mini': ['gpt-5.4-mini-2026-03-17'], 'azure::Kimi-K2.5': ['Kimi-K2.5']}
- Benchmark checks: complete OK, single_system_code OK, clean_working_tree FAILED, no_response_errors OK, no_judge_errors OK, calls_use_row_model OK, same_system_code_as_ablation OK, same_corpus_file_as_ablation OK

## Quality (ablation quality score; LISBOA minus zero-shot)

| Model | Subset | n | Zero-shot | LISBOA | Gain | 95% CI | p | r_rb |
|---|---|---:|---:|---:|---:|---|---:|---:|
| Kimi-K2.5 | all | 75 | 3.595 | 4.460 | 0.865 | [0.668, 1.055] | 1.10e-09 | 0.815 |
| Kimi-K2.5 | multi_agent | 13 | 3.750 | 4.327 | 0.577 | [0.221, 0.923] | 1.22e-02 | 0.795 |
| Kimi-K2.5 | researcher | 13 | 3.615 | 4.433 | 0.817 | [0.260, 1.288] | 1.44e-02 | 0.747 |
| Kimi-K2.5 | transport | 36 | 3.569 | 4.497 | 0.927 | [0.646, 1.188] | 8.60e-06 | 0.850 |
| Kimi-K2.5 | weather | 13 | 3.490 | 4.519 | 1.029 | [0.481, 1.519] | 6.10e-03 | 0.824 |
| gpt-5.4-mini | all | 75 | 3.580 | 4.423 | 0.843 | [0.653, 1.027] | 6.39e-10 | 0.826 |
| gpt-5.4-mini | multi_agent | 13 | 3.558 | 4.202 | 0.644 | [0.298, 0.952] | 5.37e-03 | 0.835 |
| gpt-5.4-mini | researcher | 13 | 3.740 | 4.327 | 0.587 | [-0.019, 1.163] | 1.07e-01 | 0.516 |
| gpt-5.4-mini | transport | 36 | 3.531 | 4.472 | 0.941 | [0.681, 1.181] | 4.70e-06 | 0.874 |
| gpt-5.4-mini | weather | 13 | 3.577 | 4.606 | 1.029 | [0.615, 1.413] | 1.46e-03 | 0.949 |
| gpt-5.4-mini | itinerary | 11 | 3.591 | 4.148 | 0.557 | [0.171, 0.909] | 2.15e-02 | 0.773 |
| gpt-5.4-mini | cross_domain | 2 | 3.375 | 4.500 | 1.125 | [1.000, 1.250] | 5.00e-01 | 1.000 |
| Kimi-K2.5 | itinerary | 11 | 3.818 | 4.239 | 0.420 | [0.080, 0.750] | 4.88e-02 | 0.709 |
| Kimi-K2.5 | cross_domain | 2 | 3.375 | 4.812 | 1.438 | [1.375, 1.500] | 5.00e-01 | 1.000 |

LISBOA ahead / tied / behind, per query:

- gpt-5.4-mini, all: 63 / 1 / 11 of 75
- gpt-5.4-mini, cross_domain: 2 / 0 / 0 of 2
- gpt-5.4-mini, itinerary: 9 / 0 / 2 of 11
- gpt-5.4-mini, multi_agent: 11 / 0 / 2 of 13
- gpt-5.4-mini, researcher: 9 / 0 / 4 of 13
- gpt-5.4-mini, single_domain: 52 / 1 / 9 of 62
- gpt-5.4-mini, transport: 32 / 0 / 4 of 36
- gpt-5.4-mini, weather: 11 / 1 / 1 of 13
- Kimi-K2.5, all: 63 / 1 / 11 of 75
- Kimi-K2.5, cross_domain: 2 / 0 / 0 of 2
- Kimi-K2.5, itinerary: 8 / 1 / 2 of 11
- Kimi-K2.5, multi_agent: 10 / 1 / 2 of 13
- Kimi-K2.5, researcher: 12 / 0 / 1 of 13
- Kimi-K2.5, single_domain: 53 / 0 / 9 of 62
- Kimi-K2.5, transport: 31 / 0 / 5 of 36
- Kimi-K2.5, weather: 10 / 0 / 3 of 13

## Latency and cost per response

| Model | Condition | Latency mean ± SD (s) | Median | P90 | LLM responses | Cost mean ± SD (USD) | Tokens in / out |
|---|---|---|---:|---:|---:|---|---|
| azure::gpt-5.4-mini | zero_shot | 2.01 ± 0.83 | 1.81 | 3.01 | 75/75 | 0.0012 ± 0.0008 | 154 / 233 |
| azure::gpt-5.4-mini | lisboa | 7.56 ± 7.45 | 5.79 | 12.45 | 50/75 | 0.0105 ± 0.0075 | 10167 / 640 |
| azure::Kimi-K2.5 | zero_shot | 6.37 ± 4.59 | 4.77 | 11.01 | 75/75 | 0.0028 ± 0.0018 | 159 / 889 |
| azure::Kimi-K2.5 | lisboa | 30.44 ± 31.24 | 14.93 | 80.92 | 50/75 | 0.0323 ± 0.0226 | 15683 / 7622 |
- azure::gpt-5.4-mini zero_shot: slowest tenth holds 0.206 of the time
- azure::gpt-5.4-mini lisboa: slowest tenth holds 0.346 of the time; mean by domain multi_agent 17.2, researcher 8.8, transport 5.1, weather 3.6; mean by query type cross_domain 19.5, itinerary 16.8, single_domain 5.5
- azure::Kimi-K2.5 zero_shot: slowest tenth holds 0.288 of the time
- azure::Kimi-K2.5 lisboa: slowest tenth holds 0.329 of the time; mean by domain multi_agent 69.7, researcher 48.0, transport 12.8, weather 22.4; mean by query type cross_domain 71.0, itinerary 69.5, single_domain 22.2

Run cost: responses USD 2.43, judges USD 3.95, total USD 6.38. Cost recomputed from tokens for 250 responses; largest difference USD 1.39e-17.

Zero-shot vs LISBOA per query (answers without a model call count as zero cost):

| Model | Zero-shot latency median (s) | LISBOA latency median (s) | Zero-shot cost per query (USD) | LISBOA cost per query (USD) | Cost ratio |
|---|---:|---:|---:|---:|---:|
| azure::gpt-5.4-mini | 1.81 | 5.79 | 0.00117 | 0.00700 | 5.98 |
| azure::Kimi-K2.5 | 4.77 | 14.93 | 0.00276 | 0.02152 | 7.80 |

## Worker benchmark: latency and cost per response model

| Model | Latency median (s) | P90 (s) | Model-calling responses | Cost per model-calling response (USD) | Cost per query (USD) | Tokens in / out | Errors |
|---|---:|---:|---:|---:|---:|---|---:|
| azure::Kimi-K2.5 | 0.36 | 2.89 | 8/62 | 0.01189 | 0.00153 | 15467 / 871 | 0 |
| azure::gpt-5.4-mini | 0.41 | 3.93 | 8/62 | 0.00889 | 0.00115 | 10742 / 185 | 0 |

Median latency by domain (s): azure::Kimi-K2.5: researcher 0.38, transport 0.44, weather 0.09; azure::gpt-5.4-mini: researcher 0.41, transport 0.52, weather 0.10

Benchmark run cost: responses USD 0.17, judges USD 1.36.

## End to end (LISBOA condition)

| Model | Routing: expected worker selected | Exact routing | Multi-agent coverage | Itinerary: planner | Cross-domain: planner | QA intervened | Agent source |
|---|---|---|---|---|---|---:|---|
| closed_source | 59/62 | 58/62 | 0.878 (7/13 full) | 11/11 | 0/2 | 50 | {'execution_summary': 75} |
| open_source | 59/62 | 58/62 | 0.878 (7/13 full) | 11/11 | 0/2 | 51 | {'execution_summary': 75} |

QA paths, closed_source: validated -> final-repair n=30 (QS 4.529); validated n=23 (QS 4.174); validated -> retry -> final-repair n=10 (QS 4.463); fast-weather-fact-check -> final-repair n=7 (QS 4.786); not-applicable -> final-repair n=2 (QS 4.500); fast-weather-fact-check n=1 (QS 4.875); fast-weather-fact-check -> retry -> final-repair n=1 (QS 3.875); not-applicable n=1 (QS 4.000)
Routing, closed_source: correct 62/62 (expected worker 59, boundary queries declined before any worker 3 ['W07', 'T07', 'T09']); misses none; QA worker retry 11, final repair 50 (17 by the deterministic guard alone)

QA paths, open_source: validated -> final-repair n=30 (QS 4.550); validated n=22 (QS 4.256); validated -> retry -> final-repair n=10 (QS 4.450); fast-weather-fact-check -> final-repair n=7 (QS 4.768); not-applicable -> final-repair n=2 (QS 4.438); fast-weather-fact-check n=1 (QS 4.875); fast-weather-fact-check -> retry -> final-repair n=1 (QS 3.375); not-applicable n=1 (QS 4.500); validated -> retry n=1 (QS 4.875)
Routing, open_source: correct 62/62 (expected worker 59, boundary queries declined before any worker 3 ['W07', 'T07', 'T09']); misses none; QA worker retry 12, final repair 50 (17 by the deterministic guard alone)

## Judges

| Set | Dimension | n | QWK | ICC(2,1) | ICC(2,2) | Exact | Within one | Mean abs. diff. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ablation | factual_accuracy | 300 | 0.548 | 0.549 | 0.709 | 0.320 | 0.860 | 0.833 |
| ablation | completeness | 300 | 0.732 | 0.732 | 0.845 | 0.473 | 0.907 | 0.620 |
| ablation | relevance | 300 | 0.443 | 0.444 | 0.615 | 0.637 | 0.963 | 0.403 |
| ablation | response_quality | 300 | 0.300 | 0.301 | 0.462 | 0.477 | 0.950 | 0.573 |
| benchmark | factual_accuracy | 124 | 0.323 | 0.325 | 0.490 | 0.347 | 0.903 | 0.750 |
| benchmark | tool_usage | 124 | 0.813 | 0.815 | 0.898 | 0.911 | 0.968 | 0.121 |
| benchmark | completeness | 124 | 0.749 | 0.750 | 0.857 | 0.661 | 0.992 | 0.347 |
| benchmark | relevance | 124 | 0.675 | 0.676 | 0.807 | 0.790 | 1.000 | 0.210 |
| benchmark | response_quality | 124 | 0.384 | 0.386 | 0.557 | 0.492 | 0.992 | 0.516 |

| Generator | Judges | Family | Zero-shot | LISBOA | Gain | p |
|---|---|---|---:|---:|---:|---:|
| azure::gpt-5.4-mini | both judges | both | 3.580 | 4.423 | 0.843 | 6.4e-10 |
| azure::gpt-5.4-mini | azure::gpt-5.4-mini | same family | 3.567 | 4.280 | 0.713 | 1.9e-09 |
| azure::gpt-5.4-mini | azure::Kimi-K2.5 | other family | 3.593 | 4.567 | 0.973 | 7.0e-09 |
| azure::Kimi-K2.5 | both judges | both | 3.595 | 4.460 | 0.865 | 1.1e-09 |
| azure::Kimi-K2.5 | azure::gpt-5.4-mini | other family | 3.357 | 4.277 | 0.920 | 1.0e-10 |
| azure::Kimi-K2.5 | azure::Kimi-K2.5 | same family | 3.833 | 4.643 | 0.810 | 1.3e-07 |

Benchmark, all five dimensions pooled: exact 0.640, within one 0.971, mean abs. diff. 0.389 (n=620)

Ablation, all five dimensions pooled: exact 0.505, within one 0.921, mean abs. diff. 0.580 (n=1500)

Benchmark own-family interaction: 0.061; first minus second model, by judge: {'azure::gpt-5.4-mini': 0.013, 'azure::Kimi-K2.5': -0.048}

## Itinerary constraints (judge consensus)

| Cell | Queries | Constraints | Met | Not met | Cannot assess | Disagreements | Met / all constraints | Met / unanimous assessable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| closed_source::lisboa | 11 | 58 | 55 | 2 | 0 | 1 | 0.948 | 0.965 |
| closed_source::zero_shot | 11 | 58 | 55 | 3 | 0 | 0 | 0.948 | 0.948 |
| open_source::lisboa | 11 | 58 | 53 | 4 | 0 | 1 | 0.914 | 0.930 |
| open_source::zero_shot | 11 | 58 | 55 | 0 | 0 | 3 | 0.948 | 1.000 |

Per judge:

| Cell | Judge | Constraints | Met | Not met | Cannot assess | Met / all | Met / assessable |
|---|---|---:|---:|---:|---:|---:|---:|
| closed_source::lisboa | azure::Kimi-K2.5 | 58 | 55 | 3 | 0 | 0.948 | 0.948 |
| closed_source::lisboa | azure::gpt-5.4-mini | 58 | 56 | 2 | 0 | 0.966 | 0.966 |
| closed_source::zero_shot | azure::Kimi-K2.5 | 58 | 55 | 3 | 0 | 0.948 | 0.948 |
| closed_source::zero_shot | azure::gpt-5.4-mini | 58 | 55 | 3 | 0 | 0.948 | 0.948 |
| open_source::lisboa | azure::Kimi-K2.5 | 58 | 54 | 4 | 0 | 0.931 | 0.931 |
| open_source::lisboa | azure::gpt-5.4-mini | 58 | 53 | 5 | 0 | 0.914 | 0.914 |
| open_source::zero_shot | azure::Kimi-K2.5 | 58 | 55 | 2 | 1 | 0.948 | 0.965 |
| open_source::zero_shot | azure::gpt-5.4-mini | 58 | 58 | 0 | 0 | 1.000 | 1.000 |

Inter-judge agreement on verdicts: {'judges': ['azure::Kimi-K2.5', 'azure::gpt-5.4-mini'], 'paired_verdicts': 232, 'exact_agreement': 0.9784, 'cohen_kappa': 0.7719}
