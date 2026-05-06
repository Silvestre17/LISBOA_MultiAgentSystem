# ===========================================================================
# Master Thesis - Repository Syntax Smoke Check
#   - André Filipe Gomes Silvestre, 20240502
#
# Lightweight syntax checker for a curated list of core runtime, tool,
# prompt, and evaluation modules. It complements, rather than replaces,
# the broader pytest suites.
#
# Usage:
#   > python scripts/syntax_check.py
#       Compile the curated file list and fail fast on syntax regressions.
#
# Notes:
#   - This script takes no custom parameters.
#   - For functional regressions, prefer targeted `python -m pytest ...` runs.
#   - Avoid absolute pytest paths in this workspace on Windows because the
#     OneDrive folder name contains `[` and `]`, which pytest can interpret as
#     glob characters.
# ===========================================================================

import os
import py_compile
import sys

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base)

files = [
    r"agent\agents\base.py",
    r"agent\agents\weather_agent.py",
    r"agent\agents\transport_agent.py",
    r"agent\agents\researcher_agent.py",
    r"agent\agents\supervisor.py",
    r"agent\agents\planner_agent.py",
    r"agent\agents\qa_agent.py",
    r"agent\prompts\transport.py",
    r"agent\prompts\qa.py",
    r"agent\prompts\planner.py",
    r"agent\prompts\_system_prompt.py",
    r"agent\prompts\supervisor.py",
    r"agent\prompts\researcher.py",
    r"agent\prompts\weather.py",
    r"agent\utils\response_formatter.py",
    r"agent\graph.py",
    r"agent\state.py",
    r"config.py",
    r"app.py",
    r"tools\transport_api.py",
    r"tools\ipma_api.py",
    r"tools\metrolisboa_api.py",
    r"tools\carris_api.py",
    r"tools\carrismetropolitana_api.py",
    r"tools\visitlisboa_api.py",
    "tools/dados_abertos.py",
    "tools/web_knowledge.py",
    "tools/cp_api.py",
    "tools/__init__.py",
    "eval/llm_judge.py",
    "eval/run_benchmark.py",
    "eval/run_ablation.py",
    "eval/statistical_analysis.py",
    "eval/tests/test_dataset_integrity.py",
    r"agent\llm_factory.py",
    r"agent\utils\usage_costs.py",
    r"agent\utils\langsmith_tracing.py",
    r"agent\utils\optimization.py",
    r"agent\utils\model_connection_probe.py",
    r"tools\vector_store.py",
    r"tools\location_resolver.py",
    r"tools\utils.py",
    "eval/runtime_utils.py",
    "eval/validators/transport_validator.py",
    "eval/validators/response_heuristics.py",
    "eval/tests/test_validators.py",
]

passed = 0
failed = 0
for f in files:
    full = os.path.join(base, os.path.normpath(f))
    if not os.path.exists(full):
        print(f"SKIP: {f} (not found)")
        continue
    try:
        py_compile.compile(full, doraise=True)
        print(f"OK: {f}")
        passed += 1
    except py_compile.PyCompileError as e:
        print(f"FAIL: {f} -> {e}")
        failed += 1

print(f"\n{'=' * 50}")
print(f"Passed: {passed}/{passed + failed}  Failed: {failed}/{passed + failed}")
if failed == 0:
    print("ALL SYNTAX CHECKS PASSED")
else:
    sys.exit(1)
