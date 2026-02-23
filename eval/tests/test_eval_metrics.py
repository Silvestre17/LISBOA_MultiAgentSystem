# ==========================================================================
# Master Thesis - Evaluation Metrics Unit Tests
#   - Andre Filipe Gomes Silvestre, 20240502
#
#   Unit tests for evaluate_routing_accuracy, evaluate_response_quality,
#   and evaluate_language_compliance from eval_framework.py.
#
#   Run: python -m pytest eval/tests/test_eval_metrics.py -v
# ==========================================================================

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
from eval.eval_framework import (
    evaluate_routing_accuracy,
    evaluate_response_quality,
    evaluate_language_compliance,
)


# ==========================================================================
# Tests for evaluate_routing_accuracy
# ==========================================================================


class TestRoutingAccuracy:
    """Test routing accuracy metric with 5 distinct scenarios."""

    def test_exact_match_single_agent(self):
        """Perfect routing: predicted exactly matches expected."""
        result = evaluate_routing_accuracy(["weather"], ["weather"])
        assert result["exact_match"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_exact_match_multiple_agents(self):
        """Perfect routing with multiple agents."""
        result = evaluate_routing_accuracy(
            ["weather", "transport"], ["weather", "transport"]
        )
        assert result["exact_match"] == 1.0
        assert result["f1"] == 1.0

    def test_partial_match_superset(self):
        """Predicted includes extra agents (over-routing)."""
        result = evaluate_routing_accuracy(
            ["weather", "transport"], ["weather"]
        )
        assert result["exact_match"] == 0.0
        assert result["precision"] == 0.5  # 1 correct out of 2 predicted
        assert result["recall"] == 1.0  # predicted all expected
        assert 0.6 < result["f1"] < 0.7  # 2/3 ~= 0.667

    def test_partial_match_subset(self):
        """Predicted is missing some expected agents (under-routing)."""
        result = evaluate_routing_accuracy(
            ["weather"], ["weather", "transport"]
        )
        assert result["exact_match"] == 0.0
        assert result["precision"] == 1.0  # all predicted are correct
        assert result["recall"] == 0.5  # only found 1 of 2 expected

    def test_completely_wrong(self):
        """Predicted agents are entirely different from expected."""
        result = evaluate_routing_accuracy(
            ["transport"], ["weather"]
        )
        assert result["exact_match"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_both_empty(self):
        """No agents expected and none predicted (greetings, out-of-scope)."""
        result = evaluate_routing_accuracy([], [])
        assert result["exact_match"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_predicted_empty_but_expected_not(self):
        """No agents predicted but agents were expected (missed routing)."""
        result = evaluate_routing_accuracy([], ["weather"])
        assert result["exact_match"] == 0.0
        assert result["recall"] == 0.0

    def test_expected_empty_but_predicted_some(self):
        """Agents predicted when none expected (false positive routing)."""
        result = evaluate_routing_accuracy(["transport"], [])
        assert result["exact_match"] == 0.0
        assert result["precision"] == 0.0


# ==========================================================================
# Tests for evaluate_response_quality
# ==========================================================================


class TestResponseQuality:
    """Test response quality heuristic checks."""

    def test_high_quality_response(self):
        """Response with emoji, markdown, good length, no leaks."""
        response = (
            "Hello! \U0001f324\ufe0f The weather in **Lisbon** today is sunny. "
            "Temperature is around 22\u00b0C with light winds.\n\n"
            "### Forecast\n- Tomorrow: Partly cloudy\n"
            "[More info](https://ipma.pt)"
        )
        result = evaluate_response_quality(response)
        assert result["has_content"] == 1.0
        assert result["has_emoji"] == 1.0
        assert result["has_markdown"] == 1.0
        assert result["no_tool_leaks"] == 1.0
        assert result["no_hallucinated_features"] == 1.0
        assert result["reasonable_length"] == 1.0

    def test_tool_leak_detected(self):
        """Response that exposes internal tool names should fail."""
        response = "I used get_metro_status to check the metro. All lines are running."
        result = evaluate_response_quality(response)
        assert result["no_tool_leaks"] == 0.0

    def test_hallucinated_feature_detected(self):
        """Response offering non-existent features (booking, reminders)."""
        response = (
            "Sure! I'll book a ticket for the tram 28E for you. "
            "I'll also send you a reminder before departure."
        )
        result = evaluate_response_quality(response)
        assert result["no_hallucinated_features"] == 0.0

    def test_empty_response(self):
        """Empty/None response should score 0 on all metrics."""
        result = evaluate_response_quality("")
        assert all(v == 0.0 for v in result.values())

        result_none = evaluate_response_quality(None)
        assert all(v == 0.0 for v in result_none.values())

    def test_too_short_response(self):
        """Very short response fails length and content checks."""
        result = evaluate_response_quality("OK")
        assert result["has_content"] == 0.0
        assert result["reasonable_length"] == 0.0


# ==========================================================================
# Tests for evaluate_language_compliance
# ==========================================================================


class TestLanguageCompliance:
    """Test language detection heuristic."""

    def test_portuguese_response_detected(self):
        """Portuguese response correctly identified."""
        response = "O tempo em Lisboa hoje esta bom. A temperatura e de 22 graus e nao vai chover."
        score = evaluate_language_compliance(response, "pt")
        assert score == 1.0

    def test_english_response_detected(self):
        """English response correctly identified."""
        response = "The weather in Lisbon is sunny today. You can expect temperatures around 22 degrees."
        score = evaluate_language_compliance(response, "en")
        assert score == 1.0

    def test_mixed_language_always_passes(self):
        """Mixed language queries should always pass."""
        response = "Quero ir ao Oceanario. The metro goes there via the Red Line."
        score = evaluate_language_compliance(response, "mixed")
        assert score == 1.0

    def test_wrong_language_detected(self):
        """English response when Portuguese expected should fail."""
        response = "The weather is sunny and you should bring sunscreen for the beach."
        score = evaluate_language_compliance(response, "pt")
        assert score == 0.0
