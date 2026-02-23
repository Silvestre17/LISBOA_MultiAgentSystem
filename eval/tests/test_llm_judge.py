# ==========================================================================
# Master Thesis - LLM Judge Unit Tests
#   - Andre Filipe Gomes Silvestre, 20240502
#
#   Mock-based tests for LLMJudge to validate scoring pipeline
#   without making actual API calls.
#
#   Run: python -m pytest eval/tests/test_llm_judge.py -v
# ==========================================================================

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
from unittest.mock import MagicMock, patch
from eval.llm_judge import LLMJudgeScore, LLMJudge


# ==========================================================================
# Tests for LLMJudgeScore model
# ==========================================================================


class TestLLMJudgeScore:
    """Tests for the Pydantic score model."""

    def test_composite_score_perfect(self):
        """All 5s should give composite of 5.0."""
        score = LLMJudgeScore(
            factual_accuracy=5,
            tool_usage=5,
            completeness=5,
            relevance=5,
            response_quality=5,
            reasoning="Perfect on all dimensions.",
        )
        assert score.get_composite_score() == 5.0

    def test_composite_score_minimum(self):
        """All 1s should give composite of 1.0."""
        score = LLMJudgeScore(
            factual_accuracy=1,
            tool_usage=1,
            completeness=1,
            relevance=1,
            response_quality=1,
            reasoning="Poor on all dimensions.",
        )
        assert score.get_composite_score() == 1.0

    def test_composite_score_mixed(self):
        """Mixed scores should average correctly."""
        score = LLMJudgeScore(
            factual_accuracy=5,
            tool_usage=3,
            completeness=4,
            relevance=2,
            response_quality=1,
            reasoning="Mixed results.",
        )
        expected = (5 + 3 + 4 + 2 + 1) / 5.0
        assert score.get_composite_score() == expected

    def test_score_validation_range(self):
        """Scores outside 1-5 should raise validation error."""
        with pytest.raises(Exception):
            LLMJudgeScore(
                factual_accuracy=6,
                tool_usage=1,
                completeness=1,
                relevance=1,
                response_quality=1,
                reasoning="Invalid score.",
            )

        with pytest.raises(Exception):
            LLMJudgeScore(
                factual_accuracy=0,
                tool_usage=1,
                completeness=1,
                relevance=1,
                response_quality=1,
                reasoning="Invalid score.",
            )


# ==========================================================================
# Tests for LLMJudge.evaluate (mocked LLM)
# ==========================================================================


class TestLLMJudgeEvaluate:
    """Tests for the judge evaluation pipeline using mocked LLM."""

    @patch("eval.llm_judge.LLMFactory")
    def test_evaluate_returns_correct_structure(self, mock_factory):
        """Evaluate should return dict with all expected keys."""
        # Mock the LLM to return a valid LLMJudgeScore
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = LLMJudgeScore(
            factual_accuracy=4,
            tool_usage=5,
            completeness=4,
            relevance=5,
            response_quality=4,
            reasoning="Good response with accurate facts.",
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_factory.get_llm.return_value = mock_llm

        judge = LLMJudge()
        result = judge.evaluate(
            query="What's the weather?",
            expected_facts=["Current temperature"],
            expected_tools=["get_current_weather_summary"],
            actual_tools=["get_current_weather_summary"],
            retrieved_context="{'temp': 15, 'condition': 'Sunny'}",
            response="The temperature is 15 degrees Celsius and it's sunny.",
        )

        assert "factual_accuracy" in result
        assert "tool_usage" in result
        assert "completeness" in result
        assert "relevance" in result
        assert "response_quality" in result
        assert "composite_score" in result
        assert "reasoning" in result
        assert result["composite_score"] == (4 + 5 + 4 + 5 + 4) / 5.0

    @patch("eval.llm_judge.LLMFactory")
    def test_evaluate_handles_llm_error(self, mock_factory):
        """Evaluate should return zeros when LLM fails."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("API Error: rate limit")
        mock_llm.with_structured_output.return_value = mock_structured
        mock_factory.get_llm.return_value = mock_llm

        judge = LLMJudge()
        result = judge.evaluate(
            query="Test query",
            expected_facts=[],
            expected_tools=[],
            actual_tools=[],
            retrieved_context="",
            response="Test response",
        )

        assert result["composite_score"] == 0.0
        assert "Judge Failed" in result["reasoning"]

    @patch("eval.llm_judge.LLMFactory")
    def test_evaluate_with_empty_inputs(self, mock_factory):
        """Judge should handle empty facts, tools, and context gracefully."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = LLMJudgeScore(
            factual_accuracy=5,
            tool_usage=5,
            completeness=5,
            relevance=5,
            response_quality=5,
            reasoning="Greeting handled well, no tools needed.",
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_factory.get_llm.return_value = mock_llm

        judge = LLMJudge()
        result = judge.evaluate(
            query="Hello!",
            expected_facts=[],
            expected_tools=[],
            actual_tools=[],
            retrieved_context="",
            response="Hello! Welcome to Lisbon. How can I help?",
        )

        assert result["composite_score"] == 5.0

    @patch("eval.llm_judge.LLMFactory")
    def test_evaluate_formats_prompt_correctly(self, mock_factory):
        """Check that the prompt is formatted with all fields."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = LLMJudgeScore(
            factual_accuracy=3,
            tool_usage=3,
            completeness=3,
            relevance=3,
            response_quality=3,
            reasoning="Average response.",
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_factory.get_llm.return_value = mock_llm

        judge = LLMJudge()
        judge.evaluate(
            query="Test query",
            expected_facts=["Fact A", "Fact B"],
            expected_tools=["tool_a", "tool_b"],
            actual_tools=["tool_a"],
            retrieved_context="Some context data",
            response="Some response",
        )

        # Verify invoke was called (prompt was built and sent)
        assert mock_structured.invoke.called
