# ========================================================================== 
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
# 
# Regression tests for the transport verification harness.
#
# Run from the repository root with a relative path:
#   python -m pytest tests/test_transport_verification_harness.py -q
# Useful parameters:
#   -vv                           verbose mode
#   -k reference or -k markdown   focus on one harness helper
#   -x                            stop on first failure
#   --tb=short                    shorter tracebacks
# Notes:
#   - Prefer relative paths in this workspace. Absolute pytest paths may be
#     treated as glob patterns on Windows because the folder name includes
#     `[` and `]`.
# ==========================================================================

# Required libraries:
# pip install pytest

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.run_transport_verification import (
    AuditExecutionResult,
    TransportAuditRecord,
    build_reference_result,
    normalize_audit_text,
    render_markdown_report,
    summarize_audit,
)


def test_normalize_audit_text_strips_dynamic_timestamps() -> None:
    """Audit comparisons should ignore clock noise in otherwise identical outputs."""
    normalized = normalize_audit_text(
        "📌 **Source:** Metro | **Updated:** 09:31\nAuto id: auto_deadbeef"
    )

    assert "09:31" not in normalized
    assert "auto_deadbeef" not in normalized
    assert "<time>" in normalized
    assert "auto_id" in normalized


def test_build_reference_result_prefers_deterministic_response() -> None:
    """The verification harness should use the direct deterministic response when available."""
    agent = MagicMock()
    agent._resolve_deterministic_response.return_value = "Deterministic answer"

    result = build_reference_result(agent, "Is the metro working?")

    assert result == {
        "kind": "deterministic_response",
        "path": "deterministic_response",
        "args": None,
        "raw_output": "Deterministic answer",
        "final_output": "Deterministic answer",
    }


def test_build_reference_result_invokes_tool_when_needed() -> None:
    """The verification harness should capture raw + finalized tool outputs for single-tool fast paths."""
    agent = MagicMock()
    agent._resolve_deterministic_response.return_value = None

    fake_message = MagicMock()
    fake_message.tool_calls = [{"name": "get_train_schedule", "args": {"station_name": "Entrecampos"}}]

    tool = MagicMock()
    tool.invoke.return_value = "🚆 **Departures from Entrecampos**\n🕐 **20:30** → Sintra"
    agent._get_tool_by_name.return_value = tool

    with patch("tests.run_transport_verification._build_deterministic_transport_tool_call", return_value=fake_message):
        result = build_reference_result(agent, "When are the next trains from Entrecampos?")

    assert result["kind"] == "tool_call"
    assert result["path"] == "get_train_schedule"
    assert result["args"] == {"station_name": "Entrecampos"}
    assert "Departures from Entrecampos" in result["raw_output"]
    assert result["final_output"]


def test_render_markdown_report_contains_summary_matrix_and_details() -> None:
    """Markdown rendering should include the summary matrix and per-query detail blocks."""
    records = [
        TransportAuditRecord(
            category="metro",
            query="Is the metro working?",
            language="en",
            expected_path="deterministic_response",
            reference_kind="deterministic_response",
            reference_path="deterministic_response",
            reference_args=None,
            reference_output="Metro is running normally.",
            invoke=AuditExecutionResult(status="ok", output="Metro is running normally.", similarity_to_reference=1.0),
            subgraph=AuditExecutionResult(status="ok", output="Metro is running normally.", similarity_to_reference=1.0),
            multiagent=AuditExecutionResult(status="ok", output="Metro is running normally.", similarity_to_reference=1.0),
        )
    ]

    report = render_markdown_report(records)
    summary = summarize_audit(records)

    assert "# Transport Verification Report" in report
    assert "## Quick matrix" in report
    assert "## Details" in report
    assert "Is the metro working?" in report
    assert summary["queries"] == 1
    assert summary["avg_invoke_similarity"] == 1.0
