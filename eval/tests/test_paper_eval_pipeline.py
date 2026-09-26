# ==========================================================================
# Master Thesis - Paper Evaluation Pipeline Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Deterministic checks for the RINENG revision runners and analysis:
#   checkpoint names and resumes, routing classification, and the derived
#   latency and agreement figures. No model is called.
#
#   Run from the repository root with a relative path:
#     python -m pytest eval/tests/test_paper_eval_pipeline.py -q
# ==========================================================================

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from eval import paper_eval_analysis as pea  # noqa: E402
from eval import run_ablation, run_benchmark  # noqa: E402


def test_resume_keeps_checkpoint_prefix():
    checkpoint = Path("ablation_final_20260926_081500.partial.jsonl")
    assert run_ablation._prefix_from_checkpoint(checkpoint) == "ablation_final"
    assert run_benchmark._prefix_from_checkpoint(Path("benchmark_final_20260926_120000.partial.jsonl")) == "benchmark_final"
    assert run_benchmark._prefix_from_checkpoint(Path("unexpected.partial.jsonl")) == run_benchmark.DEFAULT_OUTPUT_PREFIX


def test_benchmark_checkpoint_reuses_finished_and_retries_failed(tmp_path):
    checkpoint = tmp_path / "benchmark_final_20260926_120000.partial.jsonl"
    ok_record = {"id": "W01", "response_model": "azure::gpt-5.4-mini", "error": None, "judge_runs": [{"error": None}]}
    failed_record = {"id": "T01", "response_model": "azure::gpt-5.4-mini", "error": "timeout", "judge_runs": []}
    lines = [
        {"type": "session", "groundtruth_fingerprint": "abc", "judge_model_ids": ["a", "b"], "provenance": {"git": {"system_code_sha256": "code"}}},
        {"type": "result", "id": "W01", "response_model": "azure::gpt-5.4-mini", "record": ok_record},
        {"type": "result", "id": "T01", "response_model": "azure::gpt-5.4-mini", "record": failed_record},
    ]
    # The last line is cut, as after a crash during a write.
    checkpoint.write_text("\n".join(json.dumps(line) for line in lines) + '\n{"type": "res', encoding="utf-8")

    kept = run_benchmark._load_checkpoint(checkpoint)
    assert set(kept["completed"]) == {("W01", "azure::gpt-5.4-mini"), ("T01", "azure::gpt-5.4-mini")}
    assert kept["system_code_sha256"] == "code" and kept["judge_model_ids"] == ["a", "b"]

    retried = run_benchmark._load_checkpoint(checkpoint, retry_errors=True)
    assert set(retried["completed"]) == {("W01", "azure::gpt-5.4-mini")}
    assert retried["retried_errors"] == 1


def _lisboa_record(query_id, domain, expected_tools, selected, qa_path="validated", qa_calls=1):
    return {
        "id": query_id,
        "domain": domain,
        "expected_tools": expected_tools,
        "comparisons": {
            "closed_source": {
                "metrics": {
                    "lisboa": {
                        "execution_summary": {"selected_agents": selected, "execution_type": "single-worker", "qa_path": qa_path},
                        "agent_usage": {"qa": {"call_count": qa_calls}},
                        "scores": {"ablation_quality_score": 4.0},
                    }
                }
            }
        },
    }


def test_routing_counts_boundary_declines_as_intended():
    payload = {
        "summary": {"comparison_profile_order": ["closed_source"]},
        "ablation_results": [
            _lisboa_record("W01", "weather", ["get_weather_forecast"], ["weather"]),
            _lisboa_record("W05", "weather", [], [], qa_path="not-applicable", qa_calls=0),
            _lisboa_record("T01", "transport", ["get_metro_status"], ["researcher"]),
            _lisboa_record("T02", "transport", ["plan_route"], ["transport"], qa_path="validated -> final-repair", qa_calls=0),
        ],
    }
    section = pea.end_to_end_section(payload, None)["closed_source"]
    routing = section["routing_single_domain"]
    assert routing["expected_selected"] == 2
    assert routing["boundary_declined_ids"] == ["W05"]
    assert routing["misses"] == ["T01"]
    assert routing["correct"] == 3
    assert section["qa_final_repair"] == 1 and section["qa_final_repair_deterministic_only"] == 1


def test_latency_and_agreement_figures():
    assert pea._slowest_decile_share([1.0] * 9 + [11.0]) == 0.55
    row = pea._agreement_row([5, 4, 3, 5], [5, 3, 1, 5])
    assert row["exact_agreement"] == 0.5
    assert row["within_one_point"] == 0.75
    assert row["mean_absolute_difference"] == 0.75
