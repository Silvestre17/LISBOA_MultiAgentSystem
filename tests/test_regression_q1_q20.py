# ==========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
#
# Q1-Q20 deterministic regression fixtures for the 2026-04-15 problem set
# (see `test_queries_15.04.2026.txt` and `_/2_ProblemsPROMPT.md`).
#
# Each test pins the specific bug behavior reported in the problem prompt
# at the formatter or tool level (no live LLM calls, no network). The goal
# is not to re-test the LLM pipeline end-to-end but to make every bug we
# fixed re-discoverable as a unit test.
# ==========================================================================

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from agent.utils.response_formatter import (
    enforce_language_labels,
    final_visual_pass,
    operators_from_tool_names,
    rebuild_transport_source_line,
    reorder_tips_before_source,
    reorder_warnings_before_source,
    strip_stray_leading_enumerator,
    strip_orphan_bold_markers,
    repair_bold_time_spacing,
    linkify_phone_numbers,
    linkify_address_lines,
    resolve_output_language,
    infer_response_language,
)
from tools.transport_api import _build_ambiguity_preamble


# --------------------------------------------------------------------------
# Q1. Emoji carry-through on planner headers.
# --------------------------------------------------------------------------
def test_q1_planner_section_emoji_survives_visual_pass() -> None:
    """Planner section emojis such as ⛅ must survive final_visual_pass intact."""
    text = (
        "### Itinerário em Lisboa\n\n"
        "**⛅ Condições Meteorológicas**\n"
        "- Céu limpo.\n\n"
        "**🚇 Como Chegar**\n"
        "- Metro Azul.\n"
    )
    out = final_visual_pass(text)
    assert "⛅" in out
    assert "🚇" in out


# --------------------------------------------------------------------------
# Q2. Carris source footer must not bleed CP / Metro.
# --------------------------------------------------------------------------
def test_q2_carris_only_source_footer_after_rebuild() -> None:
    """When only Carris tools are invoked, the source footer must cite Carris only."""
    tool_names = ["carris_find_routes_between", "carris_get_next_departures"]
    ops = operators_from_tool_names(tool_names)
    assert ops == ["carris"]

    raw = (
        "🚌 Rossio → Belém via Carris.\n\n"
        "📌 **Source:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | "
        "[*Carris*](https://www.carris.pt) | [*CP*](https://www.cp.pt) | "
        "**Updated:** 14:00"
    )
    rewritten = rebuild_transport_source_line(raw, ops, language="pt")
    assert "Carris" in rewritten
    assert "Metro de Lisboa" not in rewritten
    assert "CP" not in rewritten.split("**Source:**")[-1]


# --------------------------------------------------------------------------
# Q3. Planner warnings must sit BEFORE the source footer.
# --------------------------------------------------------------------------
def test_q3_warnings_reordered_before_source() -> None:
    """⚠️ lines that appear after the 📌 source footer must move before it."""
    text = (
        "Plan body.\n\n"
        "📌 **Source:** [*IPMA*](https://www.ipma.pt) | **Updated:** 15:00\n\n"
        "⚠️ Risk of rain in the afternoon."
    )
    out = reorder_warnings_before_source(text)
    warn_idx = out.index("⚠️")
    source_idx = out.index("📌")
    assert warn_idx < source_idx


def test_q3_tips_reordered_before_source() -> None:
    """💡 tip lines that appear after the 📌 source footer must move before it."""
    text = (
        "Plan body.\n\n"
        "📌 **Source:** [*IPMA*](https://www.ipma.pt) | **Updated:** 15:00\n\n"
        "💡 Consider taking Metro Line Blue."
    )
    out = reorder_tips_before_source(text)
    tip_idx = out.index("💡")
    source_idx = out.index("📌")
    assert tip_idx < source_idx


# --------------------------------------------------------------------------
# Q4-Q5. Metro routing (deterministic linkage via location_resolver).
# These are covered by the existing test_location_resolution.py suite; we
# add a minimal regression covering the "airport to Rossio" wording shape.
# --------------------------------------------------------------------------
def test_q5_airport_rossio_language_detection_stays_english() -> None:
    """English query should not be misclassified when location names are Portuguese."""
    lang, _, _ = resolve_output_language(
        user_query="How do I get from Lisbon Airport to Rossio using the metro right now?",
        ui_default="pt",
    )
    assert lang == "en"


# --------------------------------------------------------------------------
# Q6. Same-origin PT query should stay PT.
# --------------------------------------------------------------------------
def test_q6_pt_query_language_detection_stays_pt() -> None:
    lang, _, _ = resolve_output_language(
        user_query="Como vou amanhã do Rossio ao Aeroporto de metro?",
        ui_default="en",
    )
    assert lang == "pt"


# --------------------------------------------------------------------------
# Q8. CP diacritic-insensitive station lookup ("Cais do Sodré").
# --------------------------------------------------------------------------
def test_q8_cp_search_gtfs_stop_handles_diacritics() -> None:
    """search_gtfs_stop must match 'Cais do Sodré' and 'Cais do Sodre' alike."""
    from tools import cp_api

    with_accent = cp_api.search_gtfs_stop("Cais do Sodré")
    without_accent = cp_api.search_gtfs_stop("Cais do Sodre")
    # At least one result in each case and matching name set
    assert with_accent, "Expected at least one CP stop for 'Cais do Sodré'."
    assert without_accent, "Expected at least one CP stop for 'Cais do Sodre'."
    names_with = {stop.get("stop_name", "").lower() for stop in with_accent}
    names_without = {stop.get("stop_name", "").lower() for stop in without_accent}
    assert names_with & names_without


def test_q8_cp_search_gtfs_stop_common_control_stations_return_results() -> None:
    """Known CP stations used in the April checks should all remain searchable."""
    from tools import cp_api

    expected_tokens = {
        "Amadora": "amadora",
        "Queluz": "queluz",
        "Cascais": "cascais",
    }

    for query, expected_token in expected_tokens.items():
        results = cp_api.search_gtfs_stop(query, 3)
        assert results, f"Expected at least one CP stop for '{query}'."
        assert any(expected_token in str(stop.get("stop_name", "")).lower() for stop in results)


def test_q8_plan_train_trip_handles_cais_do_sodre_to_cascais() -> None:
    """The full CP planning path should return a train trip, not a station-not-found fallback."""
    from tools.cp_api import plan_train_trip

    trip = str(plan_train_trip.invoke({"origin": "Cais do Sodré", "destination": "Cascais"}))
    normalized = trip.lower()

    assert "not found" not in normalized
    assert "cais do sodre" in normalized or "cais do sodré" in normalized
    assert "cascais" in normalized
    assert any(token in normalized for token in ["comboio", "train", "linha", "departures", "partidas"])


# --------------------------------------------------------------------------
# Q9. Seafood query in English → labels must stay English.
# --------------------------------------------------------------------------
def test_q9_pt_labels_do_not_survive_en_response() -> None:
    """enforce_language_labels must rewrite PT labels into EN when language='en'."""
    mixed = (
        "### Seafood near Tagus\n\n"
        "**Morada:** Av. Brasília, Lisbon\n"
        "**Categoria:** Restaurant, Seafood\n"
        "**Horário:** 12:00 - 23:00\n"
        "**Fonte:** [*VisitLisboa*](https://www.visitlisboa.com) | **Atualizado:** 14:00"
    )
    out = enforce_language_labels(mixed, language="en")
    assert "**Address:**" in out
    assert "**Category:**" in out
    assert "**Hours:**" in out
    assert "**Source:**" in out
    assert "**Updated:**" in out
    # and none of the PT forms survive
    assert "**Morada:**" not in out
    assert "**Categoria:**" not in out
    assert "**Fonte:**" not in out


def test_q9_en_labels_do_not_survive_pt_response() -> None:
    """Reverse direction: EN labels → PT when language='pt'."""
    mixed = (
        "**Address:** Rua X\n**Category:** Museu\n**Source:** [*VisitLisboa*]() | **Updated:** 14:00"
    )
    out = enforce_language_labels(mixed, language="pt")
    assert "**Morada:**" in out
    assert "**Categoria:**" in out
    assert "**Fonte:**" in out
    assert "**Atualizado:**" in out
    assert "**Address:**" not in out
    assert "**Source:**" not in out


# --------------------------------------------------------------------------
# Q12. French query → must be answered in English with bilingual note flag.
# --------------------------------------------------------------------------
def test_q12_fr_query_resolves_to_english_with_bilingual_note() -> None:
    lang, requires_note, detected = resolve_output_language(
        user_query="Quel temps fait-il à Lisbonne aujourd'hui?",
        ui_default="en",
    )
    assert lang == "en"
    assert requires_note is True
    assert detected == "fr"


# --------------------------------------------------------------------------
# Q13. Madeira ambiguity preamble.
# --------------------------------------------------------------------------
def test_q13_madeira_bare_triggers_ambiguity_preamble() -> None:
    out = _build_ambiguity_preamble("Rossio", "Madeira")
    assert "Ilha da Madeira" in out
    assert "Rua Humberto Madeira" in out
    assert "⚠️" in out


def test_q13_full_street_name_does_not_trigger_preamble() -> None:
    """The preamble must only fire on the bare ambiguous token."""
    assert _build_ambiguity_preamble("Rossio", "Rua Humberto Madeira") == ""
    assert _build_ambiguity_preamble("Rua Humberto Madeira", "Rossio") == ""


# --------------------------------------------------------------------------
# Q15/Q19. Orphan "1." enumerator removal.
# --------------------------------------------------------------------------
def test_q15_strip_stray_leading_enumerator_removes_orphan_one_dot() -> None:
    text = (
        "### Rossio to Belém\n\n"
        "1.\n\n"
        "**🚇 Metro route**\n"
        "- Take Line Blue.\n"
    )
    out = strip_stray_leading_enumerator(text)
    # the orphan "1." line is gone
    assert "\n1.\n" not in out
    # but the real bullet content stays
    assert "Take Line Blue" in out


# --------------------------------------------------------------------------
# Q16. Line-break normalization inside the visual pass.
# --------------------------------------------------------------------------
def test_q16_final_visual_pass_is_idempotent() -> None:
    text = (
        "**Bus Departures**\n\n"
        "- 📞 +351 213 500 115\n"
        "- 📍 Address: Rua Augusta, Lisboa\n\n"
        "**Price:** 2:  00 EUR\n"
    )
    once = final_visual_pass(text)
    twice = final_visual_pass(once)
    assert once == twice


def test_q16_blank_line_is_inserted_before_emoji_field_lines() -> None:
    text = "### Place\nOverview line\n📍 **Address:** Rua Augusta 1, Lisboa\n📞 **Phone:** +351 213 500 115"
    out = final_visual_pass(text)
    assert "Overview line\n\n📍 **Address:**" in out
    assert "Lisboa)\n\n📞 **Phone:**" in out


# --------------------------------------------------------------------------
# Q17. Phone number linkification.
# --------------------------------------------------------------------------
def test_q17_phone_number_becomes_tel_link() -> None:
    text = "- 📞 Phone: +351 213 500 115"
    out = linkify_phone_numbers(text)
    assert "tel:+351213500115" in out
    # The displayed number must remain readable (spaces preserved in the label)
    assert "+351 213 500 115" in out


# --------------------------------------------------------------------------
# Q18. Address linkification (Google Maps).
# --------------------------------------------------------------------------
def test_q18_address_becomes_google_maps_link() -> None:
    text = "- 📍 **Address:** Rua Augusta, Lisboa"
    out = linkify_address_lines(text)
    assert "google.com/maps" in out
    assert "Rua" in out


def test_q18_coordinate_pair_becomes_google_maps_link() -> None:
    text = "🗺️ **Coordinates:** (38.735010, -9.145924)"
    out = linkify_address_lines(text)
    assert "google.com/maps" in out
    assert "38.735010, -9.145924" in out


# --------------------------------------------------------------------------
# Q20. Bold-time spacing repair.
# --------------------------------------------------------------------------
def test_q20_bold_time_spacing_is_repaired() -> None:
    text = "Next train **19: 00** and another **20 :15**."
    out = repair_bold_time_spacing(text)
    assert "**19:00**" in out
    assert "**20:15**" in out


def test_q20_orphan_bold_markers_are_removed() -> None:
    text = "**⚠️ O Horário 19: 00–20:00 é apertado:**\n**"
    out = strip_orphan_bold_markers(text)
    assert out.endswith("apertado:**")
    assert "\n**" not in out


# --------------------------------------------------------------------------
# Q7, Q10, Q11, Q14, Q18b, Q19b: light sanity checks that don't duplicate
# higher-level tests but assert that the relevant helpers exist and behave
# predictably on representative inputs. These protect against silent
# regressions in the formatter wiring.
# --------------------------------------------------------------------------
def test_q7_pt_query_routes_stay_pt() -> None:
    lang, _, _ = resolve_output_language(
        user_query="O 28E está a circular a horas?",
        ui_default="en",
    )
    assert lang == "pt"


def test_q10_pharmacy_nearby_query_is_english() -> None:
    lang, _, _ = resolve_output_language(
        user_query="Where is the nearest pharmacy to Parque das Nações?",
        ui_default="pt",
    )
    assert lang == "en"


def test_q11_free_museum_query_stays_pt() -> None:
    lang, _, _ = resolve_output_language(
        user_query="Quais são os museus gratuitos este fim de semana em Lisboa?",
        ui_default="en",
    )
    assert lang == "pt"


def test_q14_fertagus_setubal_query_stays_pt() -> None:
    lang, _, _ = resolve_output_language(
        user_query="Preciso do próximo Fertagus para Setúbal e de ferry para o Barreiro agora.",
        ui_default="en",
    )
    assert lang == "pt"


def test_q18b_infer_response_language_short_input_does_not_crash() -> None:
    """Short / empty inputs must not blow up language detection."""
    assert infer_response_language(user_query="", context_text="", default="en") == "en"
    assert infer_response_language(user_query="Ok", context_text="", default="pt") == "pt"


def test_q19b_final_visual_pass_preserves_leading_enumerator_when_real_list() -> None:
    """Real enumerated lists must not be stripped by the stray-enumerator cleaner."""
    text = "Plan:\n1. Visit Belem Tower\n2. Walk to Jeronimos\n"
    out = final_visual_pass(text)
    assert "1. Visit Belem Tower" in out
    assert "2. Walk to Jeronimos" in out


# --------------------------------------------------------------------------
# Q2 extra coverage: tool-name mapping for multi-operator routes.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tools, expected",
    [
        (["get_metro_status"], ["metro"]),
        (["carris_get_stops"], ["carris"]),
        (["find_bus_routes"], ["carris_metropolitana"]),
        (["get_carris_metropolitana_alerts"], ["carris_metropolitana"]),
        (["search_cp_stations"], ["cp"]),
        (["plan_train_trip"], ["cp"]),
        (["get_transport_summary"], ["metro"]),
        (["carris_get_stops", "search_cp_stations"], ["carris", "cp"]),
        ([], []),
    ],
)
def test_q2_operators_from_tool_names_mapping(tools, expected) -> None:
    assert operators_from_tool_names(tools) == expected
