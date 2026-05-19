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
# ==========================================================================

import os
import sys
from types import SimpleNamespace

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
from agent.utils.response_formatter import canonicalize_planner_source_line, final_post_qa_guard
from eval.validators.transport_validator import (
    validate_metro_route,
    validate_response_route_facts,
    validate_station_exists,
    validate_station_on_line,
    validate_transfer_point,
)
from agent.agents.qa_agent import QualityAssuranceAgent
from agent.agents.planner_agent import (
    _build_card_based_renderer_fallback,
    _planner_response_has_markdown_contract_defects,
    _planner_response_has_transport_quality_defects,
    _planner_response_has_unrequested_sequence_stops,
    _planner_transport_bullet_is_actionable,
    _planner_response_missing_requested_movement,
    _planner_response_missing_requested_stops,
    _planner_response_violates_requested_start,
)
from agent.agents.supervisor import SupervisorAgent
from agent.agents.transport_agent import _extract_route_endpoints, _parse_route_mode_preferences
from agent.graph import MultiAgentAssistant
from agent.planning.renderer import _shared_walkable_zone
from tools import location_resolver

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


class TestQAFinalResponseAudit:
    """Deterministic QA checks that run after final response synthesis."""

    def test_requested_bus_mode_flags_metro_only_answer(self):
        """A bus request must not be accepted when the final answer is Metro-only."""
        agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)
        result = agent.assess_final_response(
            user_query="Como é que vou de autocarro entre Avenidas Novas e Campo de Ourique?",
            final_response=(
                "### 🚇 **Mobilidade em Lisboa**\n\n"
                "✅ **Resposta direta:** segue até à estação Rato.\n\n"
                "---\n\n"
                "🚇 **Destino de metro:** Rato.\n\n"
                "📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Atualizado:** 10:00"
            ),
            language="pt",
        )

        assert not result["complete"]
        assert result["needs_repair"]
        assert any("autocarro" in item for item in result["missing_data"])

    def test_requested_metro_mode_flags_cp_only_answer(self):
        """A Metro request must not be accepted when the final answer is CP-only."""
        agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)
        result = agent.assess_final_response(
            user_query="Quero ir de metro entre o Rossio e o Cais do Sodré",
            final_response=(
                "### 🚆 **Rossio → Cais do Sodré**\n\n"
                "✅ **Resposta direta:** não consegui confirmar uma ligação direta de CP suburbano.\n\n"
                "📌 **Fonte:** [*CP*](https://www.cp.pt) | **Atualizado:** 10:00"
            ),
            language="pt",
        )

        assert not result["complete"]
        assert result["needs_repair"]
        assert any("metro" in item for item in result["missing_data"])

    def test_pin_section_heading_is_not_treated_as_address_field(self):
        """A decorative/semantic pin heading must not require a Google Maps link."""
        agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)
        result = agent.assess_final_response(
            user_query="Lista atrações imperdíveis em Lisboa",
            final_response=(
                "### 📍 **Locais e atrações**\n\n"
                "✅ **Resposta direta:** aqui estão opções úteis para uma primeira visita.\n\n"
                "---\n\n"
                "- **🏛️ Sé de Lisboa**\n"
                "    - 📝 **Descrição:** Catedral histórica no centro de Lisboa.\n\n"
                "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** 10:00"
            ),
            language="pt",
        )

        critical = " ".join(result.get("critical_issues", []))
        assert "Address fields must use Google Maps links" not in critical

    def test_final_guard_repairs_route_bullet_missing_opening_bold(self):
        """Planner movement bullets must not render with dangling ``:**`` markers."""
        raw = (
            "### 🚇 **Como te deslocas**\n\n"
            "- Rua Alfa 10 → Museu Beta:** opções Carris 123, 456, 789; confirma a partida.**\n"
            "- Museu Beta → Jardim Gama:** confirma a ligação no momento.**\n\n"
            "📌 **Fonte:** [*Carris*](https://www.carris.pt) | **Atualizado:** 10:00"
        )

        repaired = final_post_qa_guard(raw, language="pt")

        assert "- **Rua Alfa 10 → Museu Beta:** opções Carris 123, 456, 789; confirma a partida." in repaired
        assert "- **Museu Beta → Jardim Gama:** confirma a ligação no momento." in repaired
        assert "- Rua Alfa 10" not in repaired
        assert "partida.**" not in repaired

    def test_final_guard_repairs_plain_transport_metric_bullets(self):
        """Transport metric lines from final repair must be render-safe."""
        raw = (
            "### 🚇 **Como te deslocas**\n"
            "- 🚌 **Rua Alfa 10 → Museu Beta:Carris 123** — direção **Terminal Beta**\n"
            "- Partida indicada:Paragem Alfa**\n"
            "- Tempo de viagem estimado:~31 min**\n"
            "- Próximos veículos indicados: 00:43 | 01:00 | 05:45**\n\n"
            "📌 **Fonte:** [*Carris*](https://www.carris.pt) | **Atualizado:** 10:00"
        )

        repaired = final_post_qa_guard(raw, language="pt")

        assert "- 🚌 **Rua Alfa 10 → Museu Beta:** Carris 123 — direção **Terminal Beta**" in repaired
        assert "- **Partida indicada:** Paragem Alfa" in repaired
        assert "- **Tempo de viagem estimado:** ~31 min" in repaired
        assert "- **Próximos veículos indicados:** 00:43 | 01:00 | 05:45" in repaired
        assert ":Carris" not in repaired
        assert "Paragem Alfa**" not in repaired

    def test_final_guard_repairs_plain_planner_movement_labels(self):
        """Final repair labels such as recommendation/note must not keep dangling bold."""
        raw = (
            "### 🚇 **Como te deslocas**\n"
            "- Deslocação recomendada:CP Comboios**, com ligação via **Paragem Alfa**.\n"
            "- Nota:** confirma a partida antes de sair.\n\n"
            "📌 **Fonte:** [*CP*](https://www.cp.pt) | **Atualizado:** 10:00"
        )

        repaired = final_post_qa_guard(raw, language="pt")

        assert "- **Deslocação recomendada:** CP Comboios, com ligação via Paragem Alfa." in repaired
        assert "- **Nota:** confirma a partida antes de sair." in repaired
        assert "CP Comboios**" not in repaired
        assert "- Nota:**" not in repaired

    def test_final_guard_collapses_duplicate_pipe_titles(self):
        """Planner item titles must not show the same place twice around a pipe."""
        raw = (
            "### 📍 **Roteiro sugerido**\n\n"
            "- **🏷️ Sé de Lisboa | Sé de Lisboa**\n"
            "- **🏷️ Granja Velha | Restaurante**\n"
            "  - 📍 **Morada:** [Largo da Sé, Lisboa](https://www.google.com/maps/search/?api=1&query=Largo+da+S%C3%A9%2C+Lisboa)\n\n"
            "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** 10:00"
        )

        repaired = final_post_qa_guard(raw, language="pt")

        assert "- **🏷️ Sé de Lisboa**" in repaired
        assert "- **🏷️ Granja Velha**" in repaired
        assert "Sé de Lisboa | Sé de Lisboa" not in repaired
        assert "Granja Velha | Restaurante" not in repaired
        assert "[*VisitLisboa Locais*]" in repaired

    def test_final_guard_localizes_transport_no_result_fragments(self):
        """PT transport answers must not keep English no-result fragments."""
        raw = (
            "### 🚌 **Opções de autocarro**\n\n"
            "#### 🚌 Carris Urban\n\n"
            "❌ **No Carris Metropolitana stops found near Campo de Ourique.**\n\n"
            "💡 **Tip:** try a more specific name, address, stop, or GPS point.\n\n"
            "📌 **Fonte:** [*Carris Metropolitana*](https://www.carrismetropolitana.pt) | **Atualizado:** 10:00"
        )

        repaired = final_post_qa_guard(raw, language="pt")

        assert "Não foram encontradas paragens da Carris Metropolitana perto de Campo de Ourique." in repaired
        assert "Dica" in repaired
        assert "Carris" in repaired
        assert "Carris Urbana" not in repaired
        assert "No Carris Metropolitana" not in repaired
        assert "Carris Urban\n" not in repaired
        assert "Carris Urban**" not in repaired
        assert "try a more specific" not in repaired

    def test_final_guard_adds_missing_material_carris_source(self):
        """Planner transport legs that use Carris must cite Carris in the final footer."""
        raw = (
            "### 🚇 **Como te deslocas**\n\n"
            "- 🚌 **Rua Alfa 10 → Museu Beta:** opções Carris: Carris 123, Carris 456.\n\n"
            "📌 **Fonte:** [*IPMA*](https://www.ipma.pt) | [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** 10:00"
        )

        repaired = final_post_qa_guard(raw, language="pt")

        assert "[*Carris*](https://www.carris.pt)" in repaired
        assert "📌 **Fonte:**" in repaired

    def test_planner_source_canonicalizer_drops_generic_carris_note_without_carris_leg(self):
        """A generic Carris confirmation note alone must not keep a Carris source."""
        raw = (
            "### 📍 **Suggested route**\n\n"
            "- **🏷️ 09:30 · Historic stop: Rua Alfa 10**\n"
            "  - 📍 **Address:** Rua Alfa 10, Lisboa\n\n"
            "### 🚇 **How to move**\n"
            "- 🚶 **Rua Alfa 10 → Museu Beta:** short walk in the historic center.\n\n"
            "### 💡 **Tips**\n"
            "- Keep 20-30 minutes of buffer between stops.\n"
            "- Carris line numbers and schedules should be confirmed at carris.pt, because GTFS data may miss very recent changes.\n\n"
            "📌 **Source:** [*Carris*](https://www.carris.pt) | [*VisitLisboa Places*](https://www.visitlisboa.com/en/places) | **Updated:** 10:00"
        )

        repaired = canonicalize_planner_source_line(raw, language="en")

        assert "Carris line numbers" not in repaired
        assert "[*Carris*](https://www.carris.pt)" not in repaired
        assert "[*VisitLisboa Places*](https://www.visitlisboa.com/en/places)" in repaired

    def test_final_audit_flags_missing_material_carris_source(self):
        """QA must reject a final answer that uses Carris data without Carris attribution."""
        agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)
        result = agent.assess_final_response(
            user_query="Cria um roteiro com deslocações por autocarro",
            final_response=(
                "### 🚇 **Como te deslocas**\n\n"
                "- 🚌 **Rua Alfa 10 → Museu Beta:** opções Carris: Carris 123, Carris 456.\n\n"
                "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** 10:00"
            ),
            language="pt",
        )

        assert not result["complete"]
        assert result["needs_repair"]
        assert any("Carris" in issue for issue in result.get("critical_issues", []))

    def test_final_guard_localizes_link_labels_in_portuguese(self):
        """PT final answers must not keep English card-link labels."""
        raw = (
            "### 🏛️ **Torre de Belém**\n\n"
            "- 🎟️ **Bilhetes:** [Buy tickets](https://bilheteira.museusemonumentos.pt/)\n"
            "- 🌐 **Website:** [Official website](https://example.org)\n\n"
            "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** 10:00"
        )

        repaired = final_post_qa_guard(raw, language="pt")

        assert "[Comprar bilhetes](" in repaired
        assert "[Página oficial](" in repaired
        assert "[Buy tickets](" not in repaired
        assert "[Official website](" not in repaired

    def test_final_audit_flags_english_link_labels_in_portuguese(self):
        """QA must catch language drift inside Markdown link text, not only bold labels."""
        agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)
        result = agent.assess_final_response(
            user_query="Mostra-me detalhes da Torre de Belém",
            final_response=(
                "### 🏛️ **Torre de Belém**\n\n"
                "- 🎟️ **Bilhetes:** [Buy tickets](https://bilheteira.museusemonumentos.pt/)\n\n"
                "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** 10:00"
            ),
            language="pt",
        )

        assert not result["complete"]
        assert any("English link labels" in issue for issue in result.get("critical_issues", []))

    def test_planner_transport_quality_flags_vague_confirm_later_text(self):
        """Planner quality must reject vague transport advice when transport context exists."""
        response = (
            "### 🚇 **Como te deslocas**\n"
            "- 🚇 **Rua Alfa 10 → Museu Beta:** usa transporte público; a ligação exata deve ser confirmada perto da hora.\n"
        )
        transport_context = (
            "### 🚇 **Ligações entre paragens do roteiro**\n"
            "- 🚌 **Rua Alfa 10 → Museu Beta:** opções Carris: Carris 123, Carris 456."
        )

        assert _planner_response_has_transport_quality_defects(
            response,
            "Cria um roteiro e inclui como me desloco",
            transport_context,
        )

    def test_final_guard_removes_invalid_carris_metropolitana_line_mix(self):
        """Carris Urban-style line IDs must not be presented as Carris Metropolitana."""
        raw = (
            "### 🚇 **Como te deslocas**\n\n"
            "- 🚌 **Alternativa:** também surgem opções de **Carris Metropolitana** nas linhas **729** e **79B**.\n"
            "- 🚌 **Rua Alfa 10 → Museu Beta:** usa **Carris 123**.\n\n"
            "📌 **Fonte:** [*Carris Metropolitana*](https://www.carrismetropolitana.pt) | [*Carris*](https://www.carris.pt) | **Atualizado:** 10:00"
        )

        repaired = final_post_qa_guard(raw, language="pt")

        assert "Carris Metropolitana" not in repaired
        assert "729" not in repaired
        assert "79B" not in repaired
        assert "[*Carris*](https://www.carris.pt)" in repaired
        assert "carrismetropolitana.pt" not in repaired

    def test_final_audit_flags_invalid_carris_metropolitana_line_mix(self):
        """QA must catch operator-boundary leaks in the final response."""
        agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)
        result = agent.assess_final_response(
            user_query="Inclui alternativas de autocarro para Belém",
            final_response=(
                "### 🚇 **Como te deslocas**\n\n"
                "- 🚌 **Alternativa:** também surgem opções de **Carris Metropolitana** nas linhas **729** e **79B**.\n\n"
                "📌 **Fonte:** [*Carris Metropolitana*](https://www.carrismetropolitana.pt) | **Atualizado:** 10:00"
            ),
            language="pt",
        )

        assert not result["complete"]
        assert any("Carris Metropolitana line identifiers" in issue for issue in result.get("critical_issues", []))

    def test_final_audit_flags_english_running_prose_in_portuguese(self):
        """QA must catch prose-level language drift, not only labels."""
        agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)
        result = agent.assess_final_response(
            user_query="Cria um roteiro em Lisboa",
            final_response=(
                "### 📅 **Plano para Belém**\n\n"
                "- **🏷️ Lisbon Cathedral**\n"
                "  - 📝 **Descrição:** Start at Lisbon Cathedral and walk downhill toward Baixa.\n\n"
                "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** 10:00"
            ),
            language="pt",
        )

        assert not result["complete"]
        assert any("English running prose" in issue for issue in result.get("critical_issues", []))

    def test_final_audit_flags_english_transport_no_result_in_portuguese(self):
        """QA must catch English transport no-result prose in Portuguese answers."""
        agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)
        result = agent.assess_final_response(
            user_query="Como vou de autocarro entre Avenidas Novas e Campo de Ourique?",
            final_response=(
                "### 🚌 **Opções de autocarro**\n\n"
                "**🚌 Carris Urban**\n\n"
                "❌ **No Carris Metropolitana stops found near Campo de Ourique.**\n\n"
                "💡 **Tip:** try a more specific name, address, stop, or GPS point.\n\n"
                "📌 **Fonte:** [*Carris Metropolitana*](https://www.carrismetropolitana.pt) | **Atualizado:** 10:00"
            ),
            language="pt",
        )

        assert not result["complete"]
        assert any("English running prose" in issue for issue in result.get("critical_issues", []))

    def test_planner_flags_missing_explicit_requested_stop(self):
        """Planner quality must reject plans that omit a user-requested anchor."""
        evidence = (
            "- **🏛️ Sé de Lisboa**\n"
            "    - 📍 **Morada:** Largo da Sé, Lisboa\n"
            "- **🏛️ Torre de Belém**\n"
            "    - 📍 **Morada:** Av. Brasília, Lisboa\n"
        )
        response = (
            "### 📅 **Plano para Belém**\n\n"
            "### 📍 **Roteiro sugerido**\n"
            "- **🏛️ Torre de Belém**\n"
        )

        assert _planner_response_missing_requested_stops(
            response,
            "Cria um roteiro que comece na Sé de Lisboa e termine na Torre de Belém",
            evidence,
        )

    def test_planner_flags_requested_start_not_first(self):
        """Planner quality must reject plans ordered against an explicit start."""
        response = (
            "### 📍 **Roteiro sugerido**\n"
            "- **🏛️ Torre de Belém**\n"
            "- **🏛️ Sé de Lisboa**\n"
        )

        assert _planner_response_violates_requested_start(
            response,
            "Cria um roteiro que comece na Sé de Lisboa e termine na Torre de Belém",
        )

    def test_planner_flags_missing_arbitrary_requested_stop(self):
        """Planner quality must generalize requested-stop checks beyond curated anchors."""
        evidence = (
            "- **📍 Rua Alfa 10**\n"
            "    - 📍 **Morada:** Rua Alfa 10, Lisboa\n"
            "- **🏛️ Museu Beta**\n"
            "    - 📍 **Morada:** Avenida Brasília, Lisboa\n"
        )
        response = (
            "### 📍 **Roteiro sugerido**\n"
            "- **🏛️ Museu Beta**\n"
        )

        assert _planner_response_missing_requested_stops(
            response,
            "Cria um roteiro que comece na Rua Alfa 10 e termine no Museu Beta",
            evidence,
        )

    def test_planner_flags_arbitrary_requested_start_not_first(self):
        """Planner ordering checks must use the requested place text, not a whitelist."""
        response = (
            "### 📍 **Roteiro sugerido**\n"
            "- **🏛️ Museu Beta**\n"
            "- **📍 Rua Alfa 10**\n"
        )

        assert _planner_response_violates_requested_start(
            response,
            "Cria um roteiro que comece na Rua Alfa 10 e termine no Museu Beta",
        )

    def test_planner_card_fallback_preserves_arbitrary_requested_sequence(self):
        """Fallback planning must not replace an explicit X-to-Y request with unrelated cards."""
        cards = [
            {
                "name": "Museu Beta",
                "category": "Museus",
                "address": "Avenida Beta 20, Lisboa",
                "description": "Espaco cultural usado como evidencia deterministica no teste.",
            },
            {
                "name": "Biblioteca Gama",
                "category": "Bibliotecas",
                "address": "Rua Gama 3, Lisboa",
                "description": "Local nao pedido que nao deve preencher um roteiro X-Y explicito.",
            },
        ]

        rendered = _build_card_based_renderer_fallback(
            user_message="Cria um roteiro curto que comece na Rua Alfa 10 e termine no Museu Beta; inclui como me desloco.",
            language="pt",
            cards=cards,
            weather_data="",
            transport_data="",
            places_data="- **Museu Beta**\n    - **Morada:** Avenida Beta 20, Lisboa",
            events_data="",
            qa_disclaimers=None,
        )

        assert "Rua Alfa 10" in rendered
        assert "Museu Beta" in rendered
        assert "Biblioteca Gama" not in rendered
        assert rendered.index("Rua Alfa 10") < rendered.index("Museu Beta")
        assert "Rua Alfa 10 → Museu Beta" in rendered

    def test_planner_flags_missing_arbitrary_requested_movement_section(self):
        """Movement checks must generalize beyond central-Lisbon to Belem examples."""
        response = (
            "### **Roteiro sugerido**\n"
            "- **Rua Alfa 10**\n"
            "- **Museu Beta**\n"
        )

        assert _planner_response_missing_requested_movement(
            response,
            "Cria um roteiro curto que comece na Rua Alfa 10 e termine no Museu Beta; inclui como me desloco.",
            "",
        )

    def test_planner_flags_wrong_movement_leg_for_arbitrary_sequence(self):
        """A movement section must belong to the explicit X-to-Y pair, not another route."""
        response = (
            "### **Roteiro sugerido**\n"
            "- **Rua Alfa 10**\n"
            "- **Museu Beta**\n\n"
            "### 🚇 **Como te deslocas**\n"
            "- 🚶 **Jardim Gama → Biblioteca Delta:** caminhada curta.\n"
        )

        assert _planner_response_missing_requested_movement(
            response,
            "Cria um roteiro curto que comece na Rua Alfa 10 e termine no Museu Beta; inclui como me desloco.",
            "- 🚶 **Jardim Gama → Biblioteca Delta:** caminhada curta.",
        )

    def test_planner_flags_base_route_without_concrete_leg_or_limitation(self):
        """A route-base sentence alone is not enough movement guidance for X-to-Y plans."""
        response = (
            "### **Roteiro sugerido**\n"
            "- **Rua Alfa 10**\n"
            "- **Museu Beta**\n\n"
            "### 🚇 **Como te deslocas**\n"
            "- 🗺️ **Percurso base:** começa em Rua Alfa 10 e segue para Museu Beta com a ligação indicada abaixo.\n"
        )

        assert _planner_response_missing_requested_movement(
            response,
            "Cria um roteiro curto que comece na Rua Alfa 10 e termine no Museu Beta; inclui como me desloco.",
            "",
        )

    def test_planner_flags_split_unrequested_movement_legs_for_strict_sequence(self):
        """Strict X-to-Y movement must not be replaced by legs through an unrequested stop."""
        response = (
            "### **Roteiro sugerido**\n"
            "- **Rua Alfa 10**\n"
            "- **Biblioteca Gama**\n"
            "- **Museu Beta**\n\n"
            "### 🚇 **Como te deslocas**\n"
            "- ⚠️ **Rua Alfa 10 -> Biblioteca Gama:** ligação não confirmada.\n"
            "- ⚠️ **Biblioteca Gama -> Museu Beta:** ligação não confirmada.\n"
        )

        assert _planner_response_missing_requested_movement(
            response,
            "Cria um roteiro curto que comece na Rua Alfa 10 e termine no Museu Beta; inclui como me desloco.",
            "",
        )

    def test_planner_fallback_filters_unrelated_movement_for_strict_sequence(self):
        """Fallback movement for X-to-Y requests must not reuse unrelated transport legs."""
        rendered = _build_card_based_renderer_fallback(
            user_message="Cria um roteiro curto que comece na Rua Alfa 10 e termine no Museu Beta; inclui como me desloco.",
            language="pt",
            cards=[
                {
                    "name": "Museu Beta",
                    "category": "Museus",
                    "address": "Avenida Beta 20, Lisboa",
                }
            ],
            weather_data="",
            transport_data="### 🚇 **Como te deslocas**\n- 🚶 **Jardim Gama → Biblioteca Delta:** caminhada curta.",
            places_data="- **Museu Beta**\n    - **Morada:** Avenida Beta 20, Lisboa",
            events_data="",
            qa_disclaimers=None,
        )

        assert "⚠️ **Rua Alfa 10 → Museu Beta:**" in rendered
        assert "Jardim Gama → Biblioteca Delta" not in rendered

    def test_planner_flags_unrequested_stop_in_strict_sequence(self):
        """Strict X-to-Y plans must not add unrelated visible stops."""
        response = (
            "### **Roteiro sugerido**\n"
            "- **Rua Alfa 10**\n"
            "- **Biblioteca Gama**\n"
            "- **Museu Beta**\n"
        )

        assert _planner_response_has_unrequested_sequence_stops(
            response,
            "Cria um roteiro curto que comece na Rua Alfa 10 e termine no Museu Beta; inclui como me desloco.",
        )

    def test_planner_flags_raw_bold_unrequested_stop_in_strict_sequence(self):
        """Strict X-to-Y stop checks must also catch raw bold route blocks without bullets."""
        response = (
            "### **Roteiro sugerido**\n"
            "**09:30 · Início: Rua Alfa 10**\n"
            "**11:00 · Paragem cultural: Biblioteca Gama**\n"
            "**12:30 · Fim: Museu Beta**\n"
        )

        assert _planner_response_has_unrequested_sequence_stops(
            response,
            "Cria um roteiro curto que comece na Rua Alfa 10 e termine no Museu Beta; inclui como me desloco.",
        )

    def test_planner_flags_unbalanced_bold_in_movement_section(self):
        """Malformed Markdown in movement instructions must force deterministic repair."""
        response = (
            "### 🚇 **Como te deslocas**\n"
            "- 🏛️ Rua Alfa 10 → Museu Beta**\n"
            "- Embarque em:Estação Alfa**\n"
        )

        assert _planner_response_has_markdown_contract_defects(response)

    def test_planner_rejects_malformed_transport_bullet(self):
        """Transport bullets with broken labels must not survive into planner fallback."""
        assert not _planner_transport_bullet_is_actionable("🏛️ Rua Alfa 10 → Museu Beta**")
        assert not _planner_transport_bullet_is_actionable("Embarque em:Estação Alfa**")

    def test_planner_flags_missing_requested_cross_zone_movement(self):
        """A center-to-Belém plan with route evidence must keep that route visible."""
        response = (
            "### 🚇 **Como te deslocas**\n"
            "- 🚶 **Padrão dos Descobrimentos → Torre de Belém:** caminhada curta.\n"
        )
        transport_context = (
            "### 🚇 **Ligações entre paragens do roteiro**\n"
            "- 🚌 **Baixa → Torre de Belém:** opções Carris: Carris 123 (~31 min), Carris 456.\n"
        )

        assert _planner_response_missing_requested_movement(
            response,
            "Começa na Sé de Lisboa, almoça na Baixa, vai à Torre de Belém e ao Padrão dos Descobrimentos; inclui como me desloco.",
            transport_context,
        )

    def test_planner_detects_event_food_route_request(self):
        """Planner transport enrichment must detect event + dinner + movement requests."""
        assert MultiAgentAssistant._planner_request_requires_event_food_route(
            "cria um plano para esta semana com um evento cultural e jantar tradicional; inclui como me desloco"
        )

    def test_planner_prioritizes_event_and_food_cards_for_route_enrichment(self):
        """Route enrichment must use generic card types, not fixed event or restaurant names."""
        cards = [
            {
                "name": "Noite de Jazz",
                "category": "Música",
                "when": "16 de maio às 21:00",
                "address": "Rua A, Lisboa",
                "url": "https://www.visitlisboa.com/en/events/noite-de-jazz",
            },
            {
                "name": "Tasca do Bairro",
                "category": "Restaurantes",
                "features": "Cozinha tradicional portuguesa",
                "address": "Rua B, Lisboa",
            },
            {
                "name": "Museu Exemplo",
                "category": "Museus",
                "address": "Rua C, Lisboa",
            },
        ]

        selected = MultiAgentAssistant._planner_event_food_route_cards(cards, [cards[2]])

        assert [card["name"] for card in selected[:2]] == ["Noite de Jazz", "Tasca do Bairro"]

    def test_planner_does_not_accept_generic_carris_as_concrete_cross_zone_leg(self):
        """Generic operator prose is not enough when a concrete route leg is requested."""
        response = (
            "### 🚇 **Como te deslocas**\n"
            "- 🚌 **Baixa → Torre de Belém:** usa Carris ou CP; confirma a ligação no momento.\n"
        )
        transport_context = (
            "### 🚇 **Ligações entre paragens do roteiro**\n"
            "- 🚌 **Baixa → Torre de Belém:** opções Carris: Carris 123 (~31 min), Carris 456.\n"
        )

        assert _planner_response_missing_requested_movement(
            response,
            "Começa na Sé de Lisboa, almoça na Baixa, vai à Torre de Belém; inclui como me desloco.",
            transport_context,
        )

    def test_final_audit_flags_missing_requested_food_stop(self):
        """Final QA must reject a plan that drops the requested lunch/food part."""
        agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)
        result = agent.assess_final_response(
            user_query="Cria um roteiro de 1 dia com almoço na Baixa e monumentos em Belém",
            final_response=(
                "### 📅 **Plano para Belém**\n\n"
                "### 📍 **Roteiro sugerido**\n"
                "- **🏛️ Torre de Belém**\n"
                "- **🏛️ Padrão dos Descobrimentos**\n\n"
                "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** 10:00"
            ),
            language="pt",
        )

        assert not result["complete"]
        assert any("refeição" in item or "restaurante" in item for item in result.get("missing_data", []))

    def test_final_audit_flags_wrong_requested_start(self):
        """Final QA must reject a plan whose first route block ignores the requested start."""
        agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)
        result = agent.assess_final_response(
            user_query="Cria um roteiro que comece na Sé de Lisboa e termine na Torre de Belém",
            final_response=(
                "### 📅 **Plano para Belém**\n\n"
                "### 📍 **Roteiro sugerido**\n"
                "- **🏛️ Torre de Belém**\n"
                "- **🏛️ Sé de Lisboa**\n\n"
                "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** 10:00"
            ),
            language="pt",
        )

        assert not result["complete"]
        assert any("primeira paragem" in item for item in result.get("missing_data", []))

    def test_final_audit_flags_missing_explicit_requested_stop(self):
        """Final QA must reject planner answers that lost an explicit requested stop."""
        agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)
        result = agent.assess_final_response(
            user_query="Cria um roteiro que comece na Sé de Lisboa e termine na Torre de Belém",
            final_response=(
                "### 📅 **Plano para Belém**\n\n"
                "### 📍 **Roteiro sugerido**\n"
                "- **🏛️ Torre de Belém**\n\n"
                "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** 10:00"
            ),
            language="pt",
        )

        assert not result["complete"]
        assert any("Sé de Lisboa" in item for item in result.get("missing_data", []))

    def test_final_audit_flags_missing_arbitrary_requested_stop(self):
        """Final QA requested-stop checks must not depend on a fixed anchor list."""
        agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)
        result = agent.assess_final_response(
            user_query="Cria um roteiro que comece na Rua Alfa 10 e termine no Museu Beta",
            final_response=(
                "### 📅 **Plano cultural**\n\n"
                "### 📍 **Roteiro sugerido**\n"
                "- **🏛️ Museu Beta**\n\n"
                "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** 10:00"
            ),
            language="pt",
        )

        assert not result["complete"]
        assert any("Rua Alfa 10" in item for item in result.get("missing_data", []))

    def test_final_audit_accepts_explicit_cross_zone_limitation(self):
        """QA should not treat a grounded unconfirmed-leg limitation as dead prose."""
        agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)
        result = agent.assess_final_response(
            user_query="Começa na Baixa, vai à Torre de Belém; inclui como me desloco.",
            final_response=(
                "### 📅 **Plano para Belém**\n\n"
                "### 📍 **Roteiro sugerido**\n"
                "- **📍 Baixa**\n"
                "- **🏛️ Torre de Belém**\n\n"
                "### 🚇 **Como te deslocas**\n"
                "- ⚠️ **Baixa → Torre de Belém:** ligação concreta não confirmada nos dados recolhidos.\n\n"
                "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** 10:00"
            ),
            language="pt",
        )

        assert not any("deslocação entre o centro" in item for item in result.get("missing_data", []))

    def test_final_audit_flags_missing_cross_zone_movement(self):
        """Final QA must reject plans that omit the requested center-to-Belém movement leg."""
        agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)
        result = agent.assess_final_response(
            user_query="Começa na Sé de Lisboa, almoça na Baixa, vai à Torre de Belém e ao Padrão dos Descobrimentos; inclui como me desloco.",
            final_response=(
                "### 📅 **Plano para Belém**\n\n"
                "### 📍 **Roteiro sugerido**\n"
                "- **🏛️ Sé de Lisboa**\n"
                "- **🍽️ Almoço na Baixa**\n"
                "- **🏛️ Torre de Belém**\n"
                "- **🏛️ Padrão dos Descobrimentos**\n\n"
                "### 🚇 **Como te deslocas**\n"
                "- 🚶 **Padrão dos Descobrimentos → Torre de Belém:** caminhada curta.\n\n"
                "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) | **Atualizado:** 10:00"
            ),
            language="pt",
        )

        assert not result["complete"]
        assert any("Belém" in item for item in result.get("missing_data", []))

    def test_walkable_zone_does_not_mark_baixa_to_belem_as_short_walk(self):
        """Same-zone walking heuristics must not bridge central Lisbon to Belém."""
        previous = SimpleNamespace(title="Baixa", details=["Almoço na Baixa"])
        current = SimpleNamespace(title="Belém", details=["Depois do almoço, segue de Baixa para Belém."])

        assert _shared_walkable_zone(previous, current) == ""


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
            "- Autocarro local disponível\n\n"
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

    def test_location_resolver_tries_user_query_before_curated_variants(self):
        """Location variants should prefer generic Nominatim search before local hints."""
        curated_key = next(iter(location_resolver._CURATED_QUERY_VARIANTS))
        variants = location_resolver._build_query_variants(curated_key)

        assert variants[0] == curated_key
        assert variants[1] == f"{curated_key}, Lisboa, Portugal"
        assert any(variant in variants[3:] for variant in location_resolver._CURATED_QUERY_VARIANTS[curated_key])

    def test_location_resolver_prefers_nominatim_before_curated_fallback(self, monkeypatch):
        """A live geocoder match should beat a matching curated gazetteer entry."""
        curated_key = next(iter(location_resolver._CURATED_LOCATION_POINTS))
        calls = []

        def fake_fetch(query):
            calls.append(query)
            if query == curated_key:
                return [
                    {
                        "lat": "38.711850",
                        "lon": "-9.129380",
                        "display_name": f"{curated_key}, Lisboa, Portugal",
                        "importance": 0.9,
                        "type": "museum",
                        "class": "tourism",
                        "address": {"city": "Lisboa"},
                    }
                ]
            return []

        monkeypatch.setattr(location_resolver, "_fetch_nominatim_results_cached", fake_fetch)

        result = location_resolver.geocode_location_name(curated_key)

        assert result is not None
        assert result["match_source"] == "nominatim"
        assert calls[0] == curated_key

    def test_location_resolver_uses_curated_only_as_fallback(self, monkeypatch):
        """Curated coordinates remain useful when Nominatim has no usable match."""
        curated_key = next(iter(location_resolver._CURATED_LOCATION_POINTS))
        monkeypatch.setattr(location_resolver, "_fetch_nominatim_results_cached", lambda query: [])
        monkeypatch.setattr(location_resolver, "_fetch_photon_results_cached", lambda query: [])

        result = location_resolver.geocode_location_name(curated_key)

        assert result is not None
        assert result["match_source"] == "curated_gazetteer"

    def test_current_transport_mode_preferences_override_alternatives(self):
        """Current explicit transport preferences must drive mode selection."""
        bus_only = _parse_route_mode_preferences("Quero ir de autocarro da Rua Alfa 10 para o Museu Beta")
        flexible = _parse_route_mode_preferences("Quero ir de metro ou autocarro da Rua Alfa 10 para o Museu Beta")

        assert bus_only["bus_only"]
        assert not bus_only["metro_only"]
        assert not flexible["bus_only"]
        assert not flexible["metro_only"]

    def test_route_endpoint_parser_removes_requested_mode_prefix(self):
        """Route endpoints must not include the user's requested transport mode."""
        assert _extract_route_endpoints(
            "Quero ir de autocarro da Rua Alfa 10 para o Museu Beta"
        ) == ("Rua Alfa 10", "Museu Beta")

    def test_supervisor_keeps_explicit_place_to_place_bus_route_on_transport(self):
        """A pure X-to-Y bus route must not be promoted into planner synthesis."""
        decision = SupervisorAgent._single_domain_override(
            "Quero ir de autocarro da Rua Alfa 10 para o Museu Beta"
        )

        assert decision is not None
        assert decision["agents"] == ["transport"]
