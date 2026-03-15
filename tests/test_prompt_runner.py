# ===========================================================================
# Master Thesis - Prompt Runner Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Regression tests for the manual prompt runner output policy.
# ===========================================================================

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scripts import run_prompts


def _smoke_args(quiet: bool = True) -> SimpleNamespace:
    """Build a minimal args object for smoke-runner unit tests."""
    return SimpleNamespace(
        limit=1,
        offset=0,
        category=None,
        provider=None,
        quiet=quiet,
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
