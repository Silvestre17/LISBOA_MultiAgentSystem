import py_compile
import os
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
    r"agent\prompts\transport.py",
    r"agent\prompts\_system_prompt.py",
    r"agent\prompts\supervisor.py",
    r"agent\prompts\researcher.py",
    r"agent\prompts\weather.py",
    r"agent\utils\response_formatter.py",
    r"agent\graph.py",
    r"agent\state.py",
    r"config.py",
    r"app_v1.py",
    r"tools\transport_api.py",
    r"tools\ipma_api.py",
    r"tools\metrolisboa_api.py",
    r"tools\carris_api.py",
    r"tools\carrismetropolitana_api.py",
    r"tools\visitlisboa_api.py",
    "tools/dados_abertos.py",
    "tools/web_knowledge.py",
    "eval/llm_judge.py",
    "eval/run_benchmark.py",
    "eval/run_ablation.py",
    "eval/eval_framework.py",
    "eval/tests/test_eval_metrics.py",
    "eval/tests/test_llm_judge.py",
    "eval/tests/test_dataset_integrity.py",
]

passed = 0
failed = 0
for f in files:
    full = os.path.join(base, f)
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

print(f"\n{'='*50}")
print(f"Passed: {passed}/{passed+failed}  Failed: {failed}/{passed+failed}")
if failed == 0:
    print("ALL SYNTAX CHECKS PASSED")
