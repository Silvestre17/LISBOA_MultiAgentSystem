# ===========================================================================
# Master Thesis - Prompt Runner Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Regression tests for the manual prompt runner output policy.
# ===========================================================================

from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scripts import run_prompts


def _smoke_args(quiet: bool = True, transcript_file: Path | None = None) -> SimpleNamespace:
    """Build a minimal args object for smoke-runner unit tests."""
    return SimpleNamespace(
        limit=1,
        offset=0,
        category=None,
        provider=None,
        quiet=quiet,
        transcript_file=transcript_file,
    )


def test_smoke_suite_skips_duplicate_final_echo_when_terminal_markdown_is_enabled(capsys) -> None:
    """Smoke runs should not print a second final-response block if the assistant already prints markdown to the terminal."""
    assistant = MagicMock()
    assistant.model_name = "Multi-Agent (test)"
    assistant.chat.return_value = "Single response body"

    with patch.object(run_prompts, "MultiAgentAssistant", return_value=assistant), patch.object(
        run_prompts,
        "SMOKE_PROMPTS",
        [("Test prompt", "en", "transport")],
    ), patch.object(
        run_prompts.Config,
        "SHOW_MARKDOWN_RESPONSE_IN_TERMINAL",
        True,
    ), patch.object(
        run_prompts.Config,
        "MODEL_PROVIDER",
        "azure",
    ):
        exit_code = run_prompts._run_smoke_suite(_smoke_args())

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "FINAL AI RESPONSE" not in captured
    assistant.chat.assert_called_once_with("Test prompt", verbose=False, language="en")


def test_smoke_suite_keeps_single_final_echo_when_terminal_markdown_is_disabled(capsys) -> None:
    """Smoke runs should still print one final-response block when the assistant summary does not include markdown."""
    assistant = MagicMock()
    assistant.model_name = "Multi-Agent (test)"
    assistant.chat.return_value = "Single response body"

    with patch.object(run_prompts, "MultiAgentAssistant", return_value=assistant), patch.object(
        run_prompts,
        "SMOKE_PROMPTS",
        [("Test prompt", "en", "transport")],
    ), patch.object(
        run_prompts.Config,
        "SHOW_MARKDOWN_RESPONSE_IN_TERMINAL",
        False,
    ), patch.object(
        run_prompts.Config,
        "MODEL_PROVIDER",
        "azure",
    ):
        exit_code = run_prompts._run_smoke_suite(_smoke_args())

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert captured.count("FINAL AI RESPONSE") == 1
    assert captured.count("Single response body") == 1


def test_smoke_suite_can_append_prompt_transcript(tmp_path, capsys) -> None:
    """Smoke runs should be able to persist the full console block to a transcript artifact."""
    assistant = MagicMock()
    assistant.model_name = "Multi-Agent (test)"
    assistant.chat.return_value = "Single response body"
    transcript_path = tmp_path / "test_queries_15.04.2026.txt"
    run_prompts._initialize_transcript(transcript_path, overwrite=True)

    with patch.object(run_prompts, "MultiAgentAssistant", return_value=assistant), patch.object(
        run_prompts,
        "SMOKE_PROMPTS",
        [("Test prompt", "en", "transport")],
    ), patch.object(
        run_prompts.Config,
        "SHOW_MARKDOWN_RESPONSE_IN_TERMINAL",
        False,
    ), patch.object(
        run_prompts.Config,
        "MODEL_PROVIDER",
        "azure",
    ):
        exit_code = run_prompts._run_smoke_suite(_smoke_args(transcript_file=transcript_path))

    captured = capsys.readouterr().out
    transcript = transcript_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert "FINAL AI RESPONSE" in captured
    assert "SMOKE TEST 1/1" in transcript
    assert "Prompt: Test prompt" in transcript
    assert "Single response body" in transcript


def test_stream_tee_supports_reconfigure_passthrough() -> None:
    """Transcript tee streams should tolerate code that reconfigures stdout/stderr."""

    class DummyStream:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def write(self, _data: str) -> int:
            return 0

        def flush(self) -> None:
            return None

        def reconfigure(self, **kwargs) -> None:
            self.calls.append(kwargs)

    primary = DummyStream()
    tee = run_prompts._StreamTee(primary, object())

    tee.reconfigure(encoding="utf-8", errors="replace")

    assert primary.calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_smoke_tool_trace_prefers_execution_summary_tool_counts(capsys) -> None:
    """Smoke footer metadata should reflect authoritative execution-summary tool counts when available."""
    run_prompts._print_smoke_tool_trace(
        messages=[],
        response="Final response",
        elapsed=12.34,
        execution_summary={
            "agent_tool_logs": {
                "researcher": [
                    {"tool_name": "search_lisbon_knowledge", "args": {"query": "foo"}},
                    {"tool_name": "search_places_attractions", "args": {"query": "bar"}},
                ],
                "qa": [
                    {"tool_name": "repair_final_response", "args": {}},
                ],
            },
            "total_tool_invocations": 3,
        },
    )

    captured = capsys.readouterr().out
    plain = re.sub(r"\x1b\[[0-9;]*m", "", captured)
    assert "[Researcher] 2 tool call(s)" in plain
    assert "[QA] 1 tool call(s)" in plain
    assert "Tools used: 3 | Latency: 12.34s" in plain
