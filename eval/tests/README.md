# LISBOA Evaluation Test Policy

This folder keeps only deterministic integrity checks for the evaluation
framework. It should not become a substitute for running the LISBOA system.

Keep tests here when they protect:

- `evaluation_groundtruth_queries.json` structure and tool references;
- validator/helper contracts used to interpret benchmark outputs;
- the paper-evaluation pipeline (`test_paper_eval_pipeline.py`): checkpoint
  names and resumes, routing classification, and derived latency and
  agreement figures, without calling a model.

Do not restore strict prompt-coverage manifests or large mocked judge suites as
the default quality gate. Benchmark and ablation quality should be checked by
running the evaluation commands, inspecting outputs, and comparing them with
real system behaviour.
