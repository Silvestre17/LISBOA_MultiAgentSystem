# ===========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
#
# Targeted tests for shared location resolution and prompt alignment.
#
# Run from the repository root with a relative path:
#   python -m pytest tests/test_location_resolution.py -q
# ===========================================================================

# Required libraries:
# pip install pytest

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.prompts.planner import get_planner_prompt
from agent.prompts.qa import get_qa_prompt
from agent.prompts.researcher import get_researcher_prompt
from tools.location_resolver import (
    build_dynamic_landmark_info,
    get_location_display_name,
)
from tools.metrolisboa_api import get_landmark_info


def test_build_dynamic_landmark_info_creates_metro_hint_for_unlisted_place() -> None:
    """Dynamic place resolution should produce a landmark-like metro hint for unlisted Lisbon places."""
    resolved_payload = {
        "success": True,
        "display_name": "Biblioteca Nacional de Portugal",
        "match_source": "nominatim",
        "scope": "lisbon_city",
        "class": "amenity",
        "type": "library",
        "nearest_metro": {
            "name": "Cidade Universitária",
            "distance_km": 0.48,
            "lines": ["amarela"],
        },
        "nearest_cp": None,
        "warnings": [],
    }

    with patch("tools.location_resolver.resolve_location_query", return_value=resolved_payload):
        info = build_dynamic_landmark_info("Biblioteca Nacional")

    assert info is not None
    assert info["display_name"] == "Biblioteca Nacional de Portugal"
    assert info["metro"] == "cidade universitária"
    assert info["line"] == "amarela"
    assert info["walking_hint_pt"] == "à biblioteca"
    assert info["walking_hint_en"] == "to the library"
    assert info["metro_walk_minutes"] >= 5


def test_build_dynamic_landmark_info_creates_cp_alternative_when_no_metro_is_nearby() -> None:
    """Dynamic place resolution should still help when only a nearby CP station is plausible."""
    resolved_payload = {
        "success": True,
        "display_name": "Praia de Carcavelos",
        "match_source": "nominatim",
        "scope": "aml",
        "class": "natural",
        "type": "beach",
        "nearest_metro": None,
        "nearest_cp": {
            "name": "Carcavelos",
            "distance_km": 0.55,
            "railways": ["cascais"],
        },
        "warnings": ["Location resolved in the AML, outside Lisbon city."],
    }

    with patch("tools.location_resolver.resolve_location_query", return_value=resolved_payload):
        info = build_dynamic_landmark_info("Praia de Carcavelos")

    assert info is not None
    assert info["train_station"] == "Carcavelos"
    assert info["alternative"] == "CP Train via Carcavelos"
    assert info["scope"] == "aml"
    assert info["train_walk_minutes"] >= 5


def test_get_location_display_name_preserves_stable_user_station_label() -> None:
    """Display labels should keep stable user station spellings like Entrecampos when canonical spacing differs."""
    resolved_payload = {
        "success": True,
        "display_name": "Entre Campos",
        "match_source": "metro_station",
    }

    with patch("tools.location_resolver.resolve_location_query", return_value=resolved_payload):
        label = get_location_display_name("Entrecampos")

    assert label == "Entrecampos"


def test_get_location_display_name_still_restores_accents_when_normalized_forms_match() -> None:
    """Display labels should still restore accents when the user input already matches the canonical station form."""
    resolved_payload = {
        "success": True,
        "display_name": "Cidade Universitária",
        "match_source": "metro_station",
    }

    with patch("tools.location_resolver.resolve_location_query", return_value=resolved_payload):
        label = get_location_display_name("Cidade Universitaria")

    assert label == "Cidade Universitária"


def test_prompt_alignment_reflects_hybrid_scope_and_accessibility_verification() -> None:
    """Planner, researcher, and QA prompts should align with hybrid Lisbon/AML scope and cautious accessibility wording."""
    planner_prompt = get_planner_prompt()
    researcher_prompt = get_researcher_prompt()
    qa_prompt_en = get_qa_prompt("en")
    qa_prompt_pt = get_qa_prompt("pt")

    assert "Lisbon city as the default scope" in planner_prompt
    assert "Metro de Lisboa só existe DENTRO da cidade de Lisboa" not in planner_prompt

    assert "AML when the intent is explicit" in researcher_prompt
    assert "LISBON CITY ONLY" not in researcher_prompt

    assert "accessibility claims that were not explicitly confirmed by the data" in qa_prompt_en
    assert "alegações de acessibilidade não confirmadas pelos dados" in qa_prompt_pt
    assert "inadequados para cadeira de rodas" not in qa_prompt_pt


def test_curated_real_world_landmarks_cover_manual_polish_examples() -> None:
    """Common real-world places mentioned during manual polish should resolve to stable curated transport anchors."""
    examples = {
        "Jardim da Estrela": {"metro": "rato", "train_station": "Santos"},
        "Biblioteca Nacional": {"metro": "entre campos", "train_station": "Entrecampos"},
        "Faculdade de Ciências": {"metro": "campo grande", "display_contains": "FCUL"},
        "Campo de Ourique": {"metro": "rato", "train_station": "Alcantara - Terra"},
        "Ajuda": {"train_station": "Belem"},
        "Oeiras": {"train_station": "Oeiras"},
    }

    for query, expectations in examples.items():
        info = get_landmark_info(query)
        assert info is not None, f"Expected curated landmark info for {query}"

        if "metro" in expectations:
            assert info.get("metro") == expectations["metro"]
        if "train_station" in expectations:
            assert info.get("train_station") == expectations["train_station"]
        if "display_contains" in expectations:
            assert expectations["display_contains"] in str(info.get("display_name", ""))


def test_curated_display_names_keep_manual_polish_examples_stable() -> None:
    """Display labels should stay clean for curated real-world examples instead of drifting to noisy geocoder labels."""
    assert get_location_display_name("Biblioteca Nacional") == "Biblioteca Nacional de Portugal"
    assert get_location_display_name("Faculdade de Ciências") == "Faculdade de Ciências da Universidade de Lisboa (FCUL)"
    assert get_location_display_name("FCUL") == "FCUL"
