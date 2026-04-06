# ==========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
#
# Transport verification harness for development testing, deterministic tool
# paths, TransportAgent invoke, TransportAgent subgraph execution, and optional
# MultiAgentAssistant chat.
#
# This script generates JSON + Markdown verification reports for a fixed
# catalogue of transport queries so scripted test scenarios and regression
# checks can be reviewed quickly during development. It is intentionally kept
# outside the formal evaluation framework in `eval/`.
#
# Run from the repository root with a relative path:
#   python scripts/run_transport_verification.py --limit 4
# ==========================================================================

# Required libraries:
# pip install langchain-core

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agents.transport_agent import (  # noqa: E402
    TransportAgent,
    _build_deterministic_transport_tool_call,
)
from agent.graph import MultiAgentAssistant  # noqa: E402
from agent.state import create_initial_state  # noqa: E402
from agent.utils.langsmith_tracing import (  # noqa: E402
    is_langsmith_tracing_opted_in,
    tracing_disabled_unless_opted_in,
)
from agent.utils.response_formatter import (  # noqa: E402
    finalize_worker_response,
    infer_response_language,
)


@dataclass(frozen=True)
class TransportAuditQuery:
    """Represents a single transport verification query."""

    category: str
    query: str
    expected_path: Optional[str] = None


@dataclass
class AuditExecutionResult:
    """Captures one execution path inside the transport verification harness."""

    status: str
    output: str
    similarity_to_reference: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class TransportAuditRecord:
    """Stores the full verification record for one transport query."""

    category: str
    query: str
    language: str
    expected_path: Optional[str]
    reference_kind: str
    reference_path: Optional[str]
    reference_args: Optional[Dict[str, Any]]
    reference_output: str
    invoke: AuditExecutionResult
    subgraph: AuditExecutionResult
    multiagent: Optional[AuditExecutionResult]


DEFAULT_AUDIT_QUERIES: List[TransportAuditQuery] = [
    TransportAuditQuery(
        category="metro",
        query="What's the current status of Lisbon metro lines right now?",
        expected_path="get_metro_status",
    ),
    TransportAuditQuery(
        category="metro",
        query="When is the next metro at Saldahna towards Odivela?",
        expected_path="deterministic_response",
    ),
    TransportAuditQuery(
        category="metro",
        query="Which metro station is nearest to GPS coordinates 38.725, -9.149?",
        expected_path="find_nearest_metro",
    ),
    TransportAuditQuery(
        category="metro",
        query="How often does the green metro line run?",
        expected_path="get_metro_frequency",
    ),
    TransportAuditQuery(
        category="carris-urban",
        query="Can you show route details for tram 28 E?",
        expected_path="carris_get_routes",
    ),
    TransportAuditQuery(
        category="carris-urban",
        query="Where is Carris route 15E right now?",
        expected_path="carris_get_realtime_vehicles",
    ),
    TransportAuditQuery(
        category="carris-urban",
        query="What are the next departures for 732 at Rossio?",
        expected_path="deterministic_response",
    ),
    TransportAuditQuery(
        category="carris-metropolitana",
        query="Show real-time Carris Metropolitana buses near Almada",
        expected_path="get_real_time_bus_positions",
    ),
    TransportAuditQuery(
        category="carris-metropolitana",
        query="What are the direct Carris Metropolitana buses from Oeiras to Amadora?",
        expected_path="find_direct_bus_lines",
    ),
    TransportAuditQuery(
        category="cp",
        query="When are the next trains from Entrecamposs?",
        expected_path="get_train_schedule",
    ),
    TransportAuditQuery(
        category="cp",
        query="How do I get from Rossio to Sintra by train?",
        expected_path="plan_train_trip",
    ),
    TransportAuditQuery(
        category="summary",
        query="How are Lisbon transports today?",
        expected_path="deterministic_response",
    ),
]


def normalize_audit_text(text: str) -> str:
    """Normalizes dynamic text fragments before comparing outputs."""
    normalized = str(text or "")
    normalized = re.sub(r"\b\d{2}:\d{2}(?::\d{2})?\b", "<TIME>", normalized)
    normalized = re.sub(r"auto_[0-9a-f]+", "auto_id", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def compute_similarity(reference: str, candidate: str) -> float:
    """Computes normalized output similarity for verification comparisons."""
    reference_norm = normalize_audit_text(reference)
    candidate_norm = normalize_audit_text(candidate)
    if not reference_norm and not candidate_norm:
        return 1.0
    if not reference_norm or not candidate_norm:
        return 0.0
    return round(SequenceMatcher(None, reference_norm, candidate_norm).ratio(), 4)


def _extract_last_message_content(result_state: Dict[str, Any]) -> str:
    """Extracts the final message content from a graph execution result."""
    messages = result_state.get("messages", [])
    if not messages:
        return ""
    last_message = messages[-1]
    if hasattr(last_message, "content"):
        return str(last_message.content or "").strip()
    return str(last_message).strip()


def build_reference_result(agent: TransportAgent, query: str) -> Dict[str, Any]:
    """Builds the deterministic reference output for a transport verification query."""
    language = infer_response_language(user_query=query, default="en")
    deterministic_response = agent._resolve_deterministic_response(
        user_message=query,
        context="",
        language=language,
    )
    if deterministic_response:
        return {
            "kind": "deterministic_response",
            "path": "deterministic_response",
            "args": None,
            "raw_output": deterministic_response,
            "final_output": deterministic_response,
        }

    tool_call = _build_deterministic_transport_tool_call(query)
    if not tool_call or not getattr(tool_call, "tool_calls", None):
        return {
            "kind": "none",
            "path": None,
            "args": None,
            "raw_output": "",
            "final_output": "",
        }

    spec = tool_call.tool_calls[0]
    tool_name = str(spec.get("name", ""))
    tool_args = dict(spec.get("args", {}))
    tool = agent._get_tool_by_name(tool_name)
    if tool is None:
        return {
            "kind": "tool_call",
            "path": tool_name,
            "args": tool_args,
            "raw_output": "",
            "final_output": "",
        }

    raw_output = str(tool.invoke(tool_args)).strip()
    final_output = finalize_worker_response(
        raw_output,
        agent_name="transport",
        user_query=query,
        language=language,
    )
    return {
        "kind": "tool_call",
        "path": tool_name,
        "args": tool_args,
        "raw_output": raw_output,
        "final_output": final_output,
    }


def run_transport_invoke(agent: TransportAgent, query: str) -> AuditExecutionResult:
    """Runs the TransportAgent invoke path for one transport verification query."""
    try:
        output = agent.invoke(query)
        return AuditExecutionResult(status="ok", output=output)
    except Exception as exc:  # pragma: no cover - defensive path
        return AuditExecutionResult(status="error", output="", notes=str(exc))


def run_transport_subgraph(agent: TransportAgent, query: str) -> AuditExecutionResult:
    """Runs the TransportAgent LangGraph subgraph path for one transport verification query."""
    try:
        state = create_initial_state()
        state["messages"].append(HumanMessage(content=query))
        state["user_context"] = {"language": infer_response_language(query, default="en")}
        graph = agent.build_subgraph()
        result_state = graph.invoke(state, config={"recursion_limit": 20})
        return AuditExecutionResult(status="ok", output=_extract_last_message_content(result_state))
    except Exception as exc:  # pragma: no cover - defensive path
        return AuditExecutionResult(status="error", output="", notes=str(exc))


def run_multiagent_chat(query: str) -> AuditExecutionResult:
    """Runs the full multi-agent chat path for one transport verification query."""
    try:
        assistant = MultiAgentAssistant()
        output = assistant.chat(
            query,
            verbose=False,
            language=infer_response_language(query, default="en"),
        )
        return AuditExecutionResult(status="ok", output=str(output).strip())
    except Exception as exc:  # pragma: no cover - defensive path
        return AuditExecutionResult(status="error", output="", notes=str(exc))


def _attach_similarity(reference_output: str, execution: AuditExecutionResult) -> AuditExecutionResult:
    """Returns a copy of an execution result with similarity filled in when possible."""
    if execution.status != "ok":
        return execution
    execution.similarity_to_reference = compute_similarity(reference_output, execution.output)
    return execution


def run_transport_verification(
    queries: Iterable[TransportAuditQuery],
    include_multiagent: bool = False,
) -> List[TransportAuditRecord]:
    """Runs the transport verification harness across the selected execution paths."""
    with tracing_disabled_unless_opted_in("LISBOA_ENABLE_CLI_LANGSMITH"):
        agent = TransportAgent()
        records: List[TransportAuditRecord] = []

        for item in queries:
            language = infer_response_language(user_query=item.query, default="en")
            reference = build_reference_result(agent, item.query)
            reference_output = str(reference.get("final_output", ""))

            invoke_result = _attach_similarity(reference_output, run_transport_invoke(agent, item.query))
            subgraph_result = _attach_similarity(reference_output, run_transport_subgraph(agent, item.query))
            multiagent_result = None
            if include_multiagent:
                multiagent_result = _attach_similarity(reference_output, run_multiagent_chat(item.query))

            records.append(
                TransportAuditRecord(
                    category=item.category,
                    query=item.query,
                    language=language,
                    expected_path=item.expected_path,
                    reference_kind=str(reference.get("kind", "none")),
                    reference_path=reference.get("path"),
                    reference_args=reference.get("args"),
                    reference_output=reference_output,
                    invoke=invoke_result,
                    subgraph=subgraph_result,
                    multiagent=multiagent_result,
                )
            )

        return records


run_transport_audit = run_transport_verification


def summarize_audit(records: Iterable[TransportAuditRecord]) -> Dict[str, Any]:
    """Builds a small summary block for the verification report."""
    record_list = list(records)
    summary: Dict[str, Any] = {
        "queries": len(record_list),
        "invoke_ok": 0,
        "subgraph_ok": 0,
        "multiagent_ok": 0,
        "avg_invoke_similarity": None,
        "avg_subgraph_similarity": None,
        "avg_multiagent_similarity": None,
    }

    def _average(values: List[float]) -> Optional[float]:
        return round(sum(values) / len(values), 4) if values else None

    invoke_scores: List[float] = []
    subgraph_scores: List[float] = []
    multiagent_scores: List[float] = []

    for record in record_list:
        if record.invoke.status == "ok":
            summary["invoke_ok"] += 1
            if record.invoke.similarity_to_reference is not None:
                invoke_scores.append(record.invoke.similarity_to_reference)
        if record.subgraph.status == "ok":
            summary["subgraph_ok"] += 1
            if record.subgraph.similarity_to_reference is not None:
                subgraph_scores.append(record.subgraph.similarity_to_reference)
        if record.multiagent and record.multiagent.status == "ok":
            summary["multiagent_ok"] += 1
            if record.multiagent.similarity_to_reference is not None:
                multiagent_scores.append(record.multiagent.similarity_to_reference)

    summary["avg_invoke_similarity"] = _average(invoke_scores)
    summary["avg_subgraph_similarity"] = _average(subgraph_scores)
    summary["avg_multiagent_similarity"] = _average(multiagent_scores)
    return summary


def render_markdown_report(records: Iterable[TransportAuditRecord]) -> str:
    """Renders a Markdown verification report with summary and per-query details."""
    record_list = list(records)
    summary = summarize_audit(record_list)
    lines = [
        "# Transport Verification Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- Queries checked: **{summary['queries']}**",
        f"- Invoke path succeeded: **{summary['invoke_ok']}**",
        f"- Subgraph path succeeded: **{summary['subgraph_ok']}**",
        f"- Multi-agent path succeeded: **{summary['multiagent_ok']}**",
        f"- Average invoke similarity: **{summary['avg_invoke_similarity']}**",
        f"- Average subgraph similarity: **{summary['avg_subgraph_similarity']}**",
        f"- Average multi-agent similarity: **{summary['avg_multiagent_similarity']}**",
        "",
        "## Quick matrix",
        "",
        "| Query | Reference path | Invoke | Subgraph | Multi-agent |",
        "| --- | --- | --- | --- | --- |",
    ]

    for record in record_list:
        invoke_cell = f"{record.invoke.status} ({record.invoke.similarity_to_reference})"
        subgraph_cell = f"{record.subgraph.status} ({record.subgraph.similarity_to_reference})"
        multiagent_cell = (
            f"{record.multiagent.status} ({record.multiagent.similarity_to_reference})"
            if record.multiagent
            else "skipped"
        )
        lines.append(
            f"| {record.query} | {record.reference_path or record.reference_kind} | {invoke_cell} | {subgraph_cell} | {multiagent_cell} |"
        )

    lines.extend(["", "## Details", ""])

    for index, record in enumerate(record_list, start=1):
        lines.extend(
            [
                f"### {index}. {record.query}",
                "",
                f"- Category: **{record.category}**",
                f"- Language: **{record.language}**",
                f"- Expected path: **{record.expected_path or 'n/a'}**",
                f"- Reference kind: **{record.reference_kind}**",
                f"- Reference path: **{record.reference_path or 'n/a'}**",
            ]
        )
        if record.reference_args:
            lines.append(f"- Reference args: `{json.dumps(record.reference_args, ensure_ascii=False, sort_keys=True)}`")
        lines.extend(
            [
                "",
                "#### Reference output",
                "",
                record.reference_output or "(empty)",
                "",
                "#### Invoke output",
                "",
                record.invoke.output or f"({record.invoke.status}: {record.invoke.notes or 'empty'})",
                "",
                "#### Subgraph output",
                "",
                record.subgraph.output or f"({record.subgraph.status}: {record.subgraph.notes or 'empty'})",
                "",
            ]
        )
        if record.multiagent is not None:
            lines.extend(
                [
                    "#### Multi-agent output",
                    "",
                    record.multiagent.output or f"({record.multiagent.status}: {record.multiagent.notes or 'empty'})",
                    "",
                ]
            )

    return "\n".join(lines).strip() + "\n"


def save_audit_report(records: Iterable[TransportAuditRecord], output_dir: Path) -> Dict[str, Path]:
    """Writes the JSON and Markdown verification artifacts to disk."""
    record_list = list(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"transport_verification_{timestamp}.json"
    md_path = output_dir / f"transport_verification_{timestamp}.md"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "summary": summarize_audit(record_list),
        "records": [asdict(record) for record in record_list],
    }

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(record_list), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for the transport verification harness."""
    parser = argparse.ArgumentParser(
        description="Run the LISBOA transport verification script used during development tests."
    )
    parser.add_argument(
        "--include-multiagent",
        action="store_true",
        help="Also run the MultiAgentAssistant chat path for every verification query.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of default verification queries (0 = all).",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Append an extra ad-hoc query to verify. Can be used multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "eval" / "results" / "transport_verification"),
        help="Directory where JSON and Markdown verification reports will be stored.",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the transport verification harness from the command line."""
    args = parse_args()
    if not is_langsmith_tracing_opted_in("LISBOA_ENABLE_CLI_LANGSMITH"):
        print("[LangSmith] CLI tracing disabled by default. Set LISBOA_ENABLE_CLI_LANGSMITH=true to opt in.")

    queries = list(DEFAULT_AUDIT_QUERIES)
    if args.limit and args.limit > 0:
        queries = queries[: args.limit]
    for extra_query in args.query:
        queries.append(
            TransportAuditQuery(category="ad-hoc", query=extra_query, expected_path=None)
        )

    records = run_transport_verification(queries, include_multiagent=args.include_multiagent)
    saved_paths = save_audit_report(records, Path(args.output_dir))
    summary = summarize_audit(records)

    print("\033[1mTransport verification complete\033[0m")
    print(f"\033[1;32mQueries:\033[0m {summary['queries']}")
    print(f"\033[1;32mInvoke OK:\033[0m {summary['invoke_ok']}")
    print(f"\033[1;32mSubgraph OK:\033[0m {summary['subgraph_ok']}")
    print(f"\033[1;32mMulti-agent OK:\033[0m {summary['multiagent_ok']}")
    print(f"\033[1;32mJSON report:\033[0m {saved_paths['json']}")
    print(f"\033[1;32mMarkdown report:\033[0m {saved_paths['markdown']}")


if __name__ == "__main__":
    main()
