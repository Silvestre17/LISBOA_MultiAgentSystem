# ==========================================================================
# Master Thesis - Validator Unit Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Pytest coverage for transport_validator.py and response_heuristics.py.
#   All tests are deterministic and require no LLM or network calls.
#
#   Run from the repository root with a relative path:
#     python -m pytest eval/tests/test_validators.py -q
#   Useful parameters:
#     -vv                         verbose mode
#     -k metro or -k heuristics   focus on one validator family
#     -x                          stop on first failure
#     --tb=short                  shorter tracebacks
#   Notes:
#     - Prefer relative paths in this workspace. Absolute pytest paths may be
#       treated as glob patterns on Windows because the folder name includes
#       `[` and `]`.
# ==========================================================================

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from eval.validators.response_heuristics import (
    check_emoji_density,
    check_hallucinated_features,
    check_language_compliance,
    check_response_length,
    check_tool_leaks,
    compare_response_contracts,
    extract_response_contract,
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

    def test_ferry_schedule_claim_is_hallucination(self):
        """Positive ferry schedule claims should be flagged because that data is not supported in-runtime."""
        r = check_hallucinated_features("The next Transtejo ferry departs at 18:10 with live updates.")
        assert r["hallucinated"]
        assert "Ferry schedule/live data" in r["flagged_claims"]

    def test_ferry_limitation_note_is_not_a_hallucination(self):
        """An honest limitation note about ferries should not be misclassified as a fake capability."""
        r = check_hallucinated_features(
            "I can't verify live ferry departures in this runtime, so please check the official operator page."
        )
        assert not r["hallucinated"]

    def test_shared_bike_live_availability_claim_is_hallucination(self):
        """Positive Gira or scooter live-availability claims should be flagged as unsupported."""
        r = check_hallucinated_features("There are 7 Gira bikes available live at the nearest dock.")
        assert r["hallucinated"]
        assert "Shared bike/scooter live availability" in r["flagged_claims"]


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

    def test_wrong_language_fails_overall(self):
        """Language mismatches should count as critical evaluation failures."""
        pt_response = "O metro está a funcionar normalmente em Lisboa."
        r = run_all_heuristics(pt_response, "en")
        assert not r["overall_pass"]
        assert "language_compliance" in r["critical_failures"]

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


class TestResponseContracts:
    """Tests for deterministic presentation-contract extraction and comparison."""

    def test_extract_response_contract_detects_sections_and_source(self):
        response = (
            "### 🌤️ Weather\n\n"
            "- Sunny today\n\n"
            "---\n\n"
            "### 🚇 Transport\n\n"
            "- Metro running normally\n\n"
            "📌 **Source:** [*IPMA*](https://www.ipma.pt) | [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Updated:** 11:10"
        )

        contract = extract_response_contract(response)

        assert contract["starts_with_title"] is True
        assert contract["has_source_line"] is True
        assert contract["top_level_headers"] == ["weather", "transport"]
        assert contract["bullet_count"] == 2

    def test_compare_response_contracts_accepts_same_structure_with_different_wording(self):
        reference = (
            "### 🌤️ Meteorologia\n\n"
            "- Céu limpo\n\n"
            "---\n\n"
            "### 🚇 Transportes\n\n"
            "- Metro em circulação normal\n\n"
            "📌 **Fonte:** [*IPMA*](https://www.ipma.pt) | **Atualizado:** 11:03"
        )
        candidate = (
            "### 🌤️ Meteorologia\n\n"
            "- Aguaceiros fracos\n\n"
            "---\n\n"
            "### 🚇 Transportes\n\n"
            "- Autocarro 728 disponível\n\n"
            "📌 **Fonte:** [*IPMA*](https://www.ipma.pt) | **Atualizado:** 11:08"
        )

        comparison = compare_response_contracts(reference, candidate)

        assert comparison["consistent"] is True
        assert comparison["issues"] == []

    def test_compare_response_contracts_flags_missing_source_and_header_drift(self):
        reference = (
            "### 🌤️ Weather\n\n"
            "- Sunny\n\n"
            "---\n\n"
            "### 🚇 Transport\n\n"
            "- Metro OK\n\n"
            "📌 **Source:** [*IPMA*](https://www.ipma.pt) | **Updated:** 11:10"
        )
        candidate = "Weather is sunny today. Metro is running normally."

        comparison = compare_response_contracts(reference, candidate)

        assert comparison["consistent"] is False
        assert "source_footer_mismatch" in comparison["issues"]
        assert "top_level_header_mismatch" in comparison["issues"]

    def test_compare_response_contracts_collapse_planner_card_variants(self):
        reference = (
            "### 📅 Itinerário Sugerido\n\n"
            "---\n\n"
            "### ☕ 14:00 · Pausa interior\n\n"
            "- Café\n\n"
            "---\n\n"
            "### 🏛️ 15:30 · Museu\n\n"
            "- Visita curta\n\n"
            "### ✨ Dicas Práticas\n\n"
            "- Leva guarda-chuva"
        )
        candidate = (
            "### 📅 Plano para a tarde\n\n"
            "---\n\n"
            "### 📍 14:30 · Chegada a Belém\n\n"
            "- Começa por um espaço interior\n\n"
            "---\n\n"
            "### 📍 16:00 · Explorar interiores\n\n"
            "- Museu recomendado\n\n"
            "### ✨ Notas Práticas\n\n"
            "- Confirma horários"
        )

        comparison = compare_response_contracts(reference, candidate)

        assert comparison["consistent"] is True

    def test_compare_response_contracts_accepts_planner_title_only_vs_title_plus_optional_sections(self):
        reference = (
            "### 📅 Itinerário para hoje\n\n"
            "- Resumo\n\n"
            "### ✨ Dicas Práticas\n\n"
            "- Confirmar horários"
        )
        candidate = (
            "### 📅 Itinerário para hoje\n\n"
            "- Resumo\n\n"
            "- Confirmar horários"
        )

        comparison = compare_response_contracts(reference, candidate)

        assert comparison["consistent"] is True

    def test_compare_response_contracts_accepts_planner_advisory_headers_and_longer_length(self):
        reference = (
            "### 📅 Itinerário Sugerido\n\n"
            + ("- Bloco detalhado de texto\n" * 10)
            + "\n### 🔎 Fontes indicadas\n\n- Confirmar horários"
        )
        candidate = (
            "### 📅 Itinerário Sugerido\n\n"
            "- Resumo compacto\n"
            "- Dica logística\n"
            "- Transporte a confirmar\n"
            "- Verificar horários\n\n"
            "### ✨ Notas Práticas\n\n"
            "- Confirmar horários"
        )

        comparison = compare_response_contracts(reference, candidate)

        assert comparison["consistent"] is True
