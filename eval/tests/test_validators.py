# ==========================================================================
# Master Thesis - Validator Unit Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Pytest coverage for transport_validator.py and response_heuristics.py.
#   All tests are deterministic and require no LLM or network calls.
#
#   Run: python -m pytest eval/tests/test_validators.py -v
# ==========================================================================

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from eval.validators.response_heuristics import (
    check_emoji_density,
    check_hallucinated_features,
    check_language_compliance,
    check_response_length,
    check_tool_leaks,
    run_all_heuristics,
)
from eval.validators.transport_validator import (
    validate_metro_route,
    validate_response_route_facts,
    validate_station_exists,
    validate_station_on_line,
    validate_transfer_point,
)

# ==========================================================================
# TransportValidator Tests
# ==========================================================================


class TestValidateStationExists:
    """Tests for validate_station_exists()."""

    def test_valid_station_exists(self):
        """Marques de Pombal is on both amarela and azul lines."""
        r = validate_station_exists("Marquês de Pombal")
        assert r["valid"], "Marquês de Pombal should exist in Metro network"
        assert "amarela" in r["lines"]
        assert "azul" in r["lines"]

    def test_fictional_station_does_not_exist(self):
        """Completely fictional station name should not be found."""
        r = validate_station_exists("Hogwarts")
        assert not r["valid"], "Hogwarts should not exist in Metro network"
        assert r["lines"] == []

    def test_airport_station_exists(self):
        """Aeroporto is a real station on vermelha."""
        r = validate_station_exists("Aeroporto")
        assert r["valid"]
        assert "vermelha" in r["lines"]

    def test_normalized_name_lookup(self):
        """Lookup should be case-insensitive."""
        r = validate_station_exists("baixa-chiado")
        assert r["valid"], "Name lookup should be case-insensitive"


class TestValidateStationOnLine:
    """Tests for validate_station_on_line()."""

    def test_aeroporto_on_vermelha(self):
        assert validate_station_on_line("Aeroporto", "vermelha")

    def test_aeroporto_not_on_azul(self):
        assert not validate_station_on_line("Aeroporto", "azul")

    def test_oriente_on_vermelha(self):
        assert validate_station_on_line("Oriente", "vermelha")

    def test_invalid_station_returns_false(self):
        assert not validate_station_on_line("Nowhere", "azul")


class TestValidateTransferPoint:
    """Tests for validate_transfer_point()."""

    def test_campo_grande_connects_amarela_verde(self):
        r = validate_transfer_point("Campo Grande", "amarela", "verde")
        assert r["valid"], "Campo Grande is a valid transfer between amarela and verde"

    def test_alameda_connects_verde_vermelha(self):
        r = validate_transfer_point("Alameda", "verde", "vermelha")
        assert r["valid"], "Alameda is a valid transfer between verde and vermelha"

    def test_rato_cannot_connect_amarela_azul(self):
        """Rato is only on amarela, so it cannot transfer to azul."""
        r = validate_transfer_point("Rato", "amarela", "azul")
        assert not r["valid"], "Rato only serves amarela"

    def test_rossio_cannot_connect_amarela_vermelha(self):
        r = validate_transfer_point("Rossio", "amarela", "vermelha")
        assert not r["valid"]


class TestValidateMetroRoute:
    """Tests for validate_metro_route()."""

    def test_odivelas_to_telheiras_via_campo_grande(self):
        """Dataset case T14: Odivelas (amarela) -> Campo Grande -> Telheiras (verde)."""
        r = validate_metro_route(
            origin="Odivelas",
            destination="Telheiras",
            claimed_transfer="Campo Grande",
            claimed_transfer_lines=("amarela", "verde"),
        )
        assert r["route_valid"], f"Should be valid: {r}"

    def test_baixa_chiado_to_aeroporto_via_alameda(self):
        """Dataset case T02: Baixa-Chiado (verde) -> Alameda -> Aeroporto (vermelha)."""
        r = validate_metro_route(
            origin="Baixa-Chiado",
            destination="Aeroporto",
            claimed_transfer="Alameda",
            claimed_transfer_lines=("verde", "vermelha"),
        )
        assert r["route_valid"], f"Should be valid: {r}"

    def test_invalid_transfer_makes_route_invalid(self):
        """Rossio cannot connect amarela with vermelha."""
        r = validate_metro_route(
            origin="Rato",
            destination="Aeroporto",
            claimed_transfer="Rossio",
            claimed_transfer_lines=("amarela", "vermelha"),
        )
        assert not r["route_valid"], "Bad transfer at Rossio should invalidate route"

    def test_direct_route_single_line(self):
        """Cais do Sodré to Telheiras on verde (no transfer needed)."""
        r = validate_metro_route(
            origin="Cais do Sodré",
            destination="Telheiras",
            claimed_line="verde",
        )
        assert r["route_valid"], f"Direct route on verde should be valid: {r}"


class TestValidateResponseRouteFacts:
    """Tests for validate_response_route_facts()."""

    def test_response_mentioning_correct_stations_scores_high(self):
        response = (
            "Take the Green Line from Baixa-Chiado and transfer at Alameda "
            "to the Red Line to reach Aeroporto."
        )
        r = validate_response_route_facts(
            response_text=response,
            expected_facts=["Green Line", "Alameda", "Red Line", "Aeroporto"],
        )
        assert r["facts_score"] >= 0.5

    def test_empty_facts_returns_perfect_score(self):
        r = validate_response_route_facts(response_text="Any response.", expected_facts=[])
        assert r["facts_score"] == 1.0

    def test_wrong_stations_scores_low(self):
        response = "Take the Blue Line directly."
        r = validate_response_route_facts(
            response_text=response,
            expected_facts=["Aeroporto", "Alameda", "vermelha"],
        )
        assert r["facts_score"] < 0.5


# ==========================================================================
# ResponseHeuristics Tests
# ==========================================================================


class TestCheckToolLeaks:
    """Tests for check_tool_leaks()."""

    def test_clean_response_has_no_leaks(self):
        r = check_tool_leaks("The weather in Lisbon is sunny and 22 degrees Celsius.")
        assert not r["leaked"]
        assert r["leaked_items"] == []

    def test_tool_function_name_detected(self):
        r = check_tool_leaks("Based on get_weather_forecast, it will be sunny.")
        assert r["leaked"]
        assert any("get_weather_forecast" in t for t in r["leaked_items"])

    def test_api_artifact_detected(self):
        r = check_tool_leaks('The result is ToolMessage(content="sunny").')
        assert r["leaked"]

    def test_bracket_artifact_detected(self):
        r = check_tool_leaks("[Tool: get_metro_status] The metro is running normally.")
        assert r["leaked"]


class TestCheckResponseLength:
    """Tests for check_response_length()."""

    def test_very_short_response_fails(self):
        r = check_response_length("OK.")
        assert not r["acceptable"]

    def test_normal_length_response_passes(self):
        r = check_response_length(
            "The weather in Lisbon today is sunny with temperatures around 22 degrees Celsius. "
            "No rain is expected for the next three days."
        )
        assert r["acceptable"]

    def test_single_word_fails(self):
        r = check_response_length("Yes")
        assert not r["acceptable"]


class TestCheckLanguageCompliance:
    """Tests for check_language_compliance()."""

    def test_portuguese_response_for_pt_query(self):
        pt = "O tempo em Lisboa hoje é ensolarado com temperaturas de 22 graus."
        r = check_language_compliance(pt, "pt")
        assert r["compliant"]

    def test_english_response_for_en_query(self):
        en = "The weather in Lisbon today is sunny with temperatures around 22 degrees."
        r = check_language_compliance(en, "en")
        assert r["compliant"]

    def test_english_response_for_pt_query_fails(self):
        en = "The weather in Lisbon today is sunny with temperatures around 22 degrees."
        r = check_language_compliance(en, "pt")
        assert not r["compliant"]

    def test_mixed_language_query_always_passes(self):
        r = check_language_compliance("Anything here.", "mixed")
        assert r["compliant"]


class TestCheckHallucinatedFeatures:
    """Tests for check_hallucinated_features()."""

    def test_booking_claim_is_hallucination(self):
        halluc = "I can book you a table at the restaurant for tomorrow evening."
        r = check_hallucinated_features(halluc)
        assert r["hallucinated"]

    def test_valid_metro_response_no_hallucination(self):
        valid = "The Lisbon Metro has four lines. I recommend taking the Green line to Oriente."
        r = check_hallucinated_features(valid)
        assert not r["hallucinated"]

    def test_purchase_tickets_claim_is_hallucination(self):
        r = check_hallucinated_features("I can purchase your tickets directly here.")
        assert r["hallucinated"]


class TestCheckEmojiDensity:
    """Tests for check_emoji_density()."""

    def test_clean_text_passes(self):
        r = check_emoji_density("The metro is running normally. Please check the schedule.")
        assert r["acceptable"]

    def test_heavy_emoji_usage_fails(self):
        heavy = "🌞🌤️☀️ The weather is great! 🎉🎊🥳 Visit the museums! 🏛️🖼️🎨"
        r = check_emoji_density(heavy, max_ratio=0.05)
        assert not r["acceptable"]


class TestRunAllHeuristics:
    """Integration tests for run_all_heuristics()."""

    def test_clean_response_overall_pass(self):
        clean = (
            "The weather in Lisbon today is sunny with temperatures around 22 degrees Celsius. "
            "No rain is expected for the next three days."
        )
        r = run_all_heuristics(clean, "en")
        assert r["overall_pass"], f"Clean response should pass: {r['critical_failures']}"

    def test_response_with_tool_leak_fails(self):
        """Tool name in response must cause overall failure."""
        bad = "According to get_metro_status, the metro is running normally."
        r = run_all_heuristics(bad, "en")
        assert not r["overall_pass"]
        assert "tool_leaks" in r["critical_failures"]

    def test_too_short_response_fails(self):
        r = run_all_heuristics("OK.", "en")
        assert not r["overall_pass"]
        assert "response_length" in r["critical_failures"]

    def test_aggregate_multiple_failures(self):
        """Tool leak + too short: both critical failures should appear."""
        bad = "get_metro_status OK."
        r = run_all_heuristics(bad, "en")
        assert not r["overall_pass"]
        assert "tool_leaks" in r["critical_failures"]
