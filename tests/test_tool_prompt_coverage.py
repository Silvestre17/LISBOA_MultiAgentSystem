# ===========================================================================
# Master Thesis - Strict Live Tool Coverage Tests
#   - André Filipe Gomes Silvestre, 20240502
#
# Exhaustive live prompt coverage for all exported worker-agent tools.
# This suite is intentionally slow and should remain isolated from fast tests.
#
# Run from the repository root with a relative path:
#   python -m pytest tests/test_tool_prompt_coverage.py -q --run-live -m "live and coverage"
# Useful parameters:
#   -s              show live tool/log output
#   -vv             verbose mode
#   -x              stop on first failure
#   --tb=short      shorter tracebacks
# Notes:
#   - This suite is slow and uses real services.
#   - Prefer relative paths in this workspace. Absolute pytest paths may be
#     treated as glob patterns on Windows because the folder name includes
#     `[` and `]`.
# ===========================================================================

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import ToolMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agents.researcher_agent import ResearcherAgent
from agent.agents.transport_agent import TransportAgent
from agent.agents.weather_agent import WeatherAgent
from eval.runtime_utils import (
    build_model_manifest,
    build_results_output_path,
    categorize_error,
    compute_tool_metrics,
    compute_tool_registry_fingerprint,
    fingerprint_payload,
    summarize_error_categories,
)

AGENT_CLASSES = {
    "weather": WeatherAgent,
    "transport": TransportAgent,
    "researcher": ResearcherAgent,
}
COVERAGE_MANIFEST_PATH = PROJECT_ROOT / "tests" / "fixtures" / "tool_coverage_manifest.json"



def _run_preconfigured_agent(agent: Any, query: str) -> tuple[str, list[str], str, float, str | None]:
    """Run a preconfigured worker agent and capture tools/context like the benchmark."""
    start_time = time.time()
    tools_called: list[str] = []
    retrieved_context_blocks: list[str] = []
    final_response = ""
    error: str | None = None

    try:
        graph = agent.build_subgraph()
        result = graph.invoke({"messages": [("user", query)]})

        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tools_called.append(tool_call["name"])
            elif isinstance(msg, ToolMessage):
                tool_name = msg.name if getattr(msg, "name", None) else "unknown_tool"
                retrieved_context_blocks.append(f"[{tool_name}] returned:\n{msg.content}")

        final_msg = result.get("messages", [])[-1]
        final_response = getattr(final_msg, "content", "")
    except Exception as exc:
        error = str(exc)
        final_response = f"Execution Error: {error}"

    latency = time.time() - start_time
    retrieved_context = "\n---\n".join(retrieved_context_blocks)
    return final_response, tools_called, retrieved_context, latency, error


@pytest.mark.live
@pytest.mark.coverage
def test_all_exported_tools_are_used_at_least_once(
    strict_live_environment,
    coverage_manifest,
    exported_tool_names,
    agent_model_configs,
):
    """Run the strict live manifest and assert full exported tool coverage."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    coverage_records: list[dict[str, Any]] = []
    agents = {}
    for domain, agent_class in AGENT_CLASSES.items():
        agent = agent_class()
        config = agent_model_configs[domain]
        agent.init_llm(
            provider=config["provider"],
            model=config["model"],
            temperature=config["temperature"],
        )
        agents[domain] = agent

    aggregate_tools: set[str] = set()
    failures: list[str] = []
    response_model_configs = {
        domain: build_model_manifest(
            config["provider"],
            config["model"],
            config.get("temperature"),
        )
        for domain, config in agent_model_configs.items()
    }

    for item in coverage_manifest:
        response, tools_used, retrieved_context, latency, error = _run_preconfigured_agent(
            agents[item["domain"]],
            item["query"],
        )
        tools_set = set(tools_used)
        expected_set = set(item["expected_tools"])
        aggregate_tools.update(tools_set)

        coverage_pass = error is None and expected_set.issubset(tools_set)
        coverage_records.append({
            "id": item["id"],
            "domain": item["domain"],
            "language": item.get("language", "en"),
            "query": item["query"],
            "response_model": response_model_configs[item["domain"]]["model_id"],
            "response_model_config": response_model_configs[item["domain"]],
            "latency_s": round(latency, 3),
            "error": error,
            "error_type": categorize_error(error),
            "response": response,
            "tools_used": tools_used,
            "expected_tools": item["expected_tools"],
            "tool_metrics": compute_tool_metrics(item["expected_tools"], tools_used),
            "retrieved_context": retrieved_context,
            "coverage_pass": coverage_pass,
        })

        if error is not None:
            failures.append(f"{item['id']} raised error: {error}")
            continue

        if not expected_set.issubset(tools_set):
            failures.append(
                f"{item['id']} expected {sorted(expected_set)} but observed {sorted(tools_set)}"
            )

    missing_tools = sorted(exported_tool_names - aggregate_tools)
    output_path = build_results_output_path("coverage", "coverage_results", timestamp)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "coverage_metadata": {
                    "manifest_path": str(COVERAGE_MANIFEST_PATH),
                    "manifest_count": len(coverage_manifest),
                    "manifest_fingerprint": fingerprint_payload(coverage_manifest),
                    "tool_registry_count": len(exported_tool_names),
                    "tool_registry_fingerprint": compute_tool_registry_fingerprint(exported_tool_names),
                    "response_models": {
                        domain: manifest["model_id"]
                        for domain, manifest in response_model_configs.items()
                    },
                    "response_model_configs": response_model_configs,
                    "evaluation_model": None,
                    "environment_validation": strict_live_environment,
                    "timestamp": datetime.now().isoformat(),
                    "real_services": True,
                    "output_directory": str(output_path.parent),
                },
                "summary": {
                    "total_prompts": len(coverage_records),
                    "successful_prompts": sum(1 for record in coverage_records if record["coverage_pass"]),
                    "failed_prompts": sum(1 for record in coverage_records if not record["coverage_pass"]),
                    "full_registry_coverage_met": not missing_tools,
                    "missing_tools": missing_tools,
                    "avg_tool_f1": round(
                        sum(record["tool_metrics"]["tool_f1"] for record in coverage_records) / len(coverage_records),
                        3,
                    ) if coverage_records else 0,
                    "error_categories": summarize_error_categories(coverage_records),
                },
                "coverage_results": coverage_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    problems: list[str] = []

    if failures:
        problems.append(
            "Per-case coverage mismatches:\n" + "\n".join(failures[:25]) + f"\n\nCoverage artefact: {output_path}"
        )
    if missing_tools:
        problems.append(
            "Missing tools across strict live suite:\n" + "\n".join(missing_tools) + f"\n\nCoverage artefact: {output_path}"
        )

    assert not problems, "\n\n".join(problems)
