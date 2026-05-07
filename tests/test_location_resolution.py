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
    AML_BOUNDS,
    _build_query_variants,
    _fetch_nominatim_results_cached,
    build_location_ambiguity_preamble,
    build_dynamic_landmark_info,
    geocode_location_name,
    get_location_display_name,
    resolve_location_query,
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


def test_build_location_ambiguity_preamble_flags_bare_madeira() -> None:
    """Bare Madeira should surface island-vs-Lisbon ambiguity before routing continues."""
    preamble = build_location_ambiguity_preamble("Rossio", "Madeira", language="pt")

    assert "Ambiguidade" in preamble
    assert "Ilha da Madeira" in preamble
    assert "Rua Humberto Madeira" in preamble


def test_build_location_ambiguity_preamble_ignores_explicit_madeira_address() -> None:
    """Explicit Madeira street/address wording should not trigger bare-name disambiguation."""
    preamble = build_location_ambiguity_preamble(
        "Rossio",
        "Avenida da Ilha da Madeira",
        language="pt",
    )

    assert preamble == ""


def test_build_location_ambiguity_preamble_does_not_flag_marques() -> None:
    """Marquês should not be treated as ambiguous because station and roundabout share the same practical area."""
    preamble = build_location_ambiguity_preamble("Marquês", "Belém", language="pt")

    assert preamble == ""


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


def test_city_centre_aliases_resolve_to_stable_central_lisbon_queries() -> None:
    """Vague city-centre queries should start from stable central Lisbon anchors instead of noisy OSM matches."""
    variants = _build_query_variants("centre of Lisbon")

    assert variants[0] == "Rossio, Lisboa, Portugal"
    assert "Baixa-Chiado, Lisboa, Portugal" in variants
    assert get_location_display_name("centre of Lisbon") == "Rossio"
    assert get_location_display_name("center") == "Rossio"
    assert get_location_display_name("centre") == "Rossio"


def test_curated_gazetteer_resolves_common_museums_without_network() -> None:
    """High-frequency Lisbon venues should not depend on Nominatim availability."""
    museum_queries = {
        "Museu Nacional do Azulejo": "Museu Nacional do Azulejo",
        "Museu Calouste Gulbenkian": "Museu Calouste Gulbenkian",
        "MAAT - Museu de Arte, Arquitetura e Tecnologia": "MAAT",
        "Museu do Fado": "Museu do Fado",
        "National Museum of Ancient Art": "Museu Nacional de Arte Antiga",
        "Carris Museum (Public Transport Museum)": "Carris Museum",
        "Museu Coleção Berardo": "Museu Coleção Berardo",
        "National Coach Museum": "Museu Nacional dos Coches",
        "Museu do Oriente": "Museu do Oriente",
        "Museu Nacional de História Natural e da Ciência": "Museu Nacional de História Natural",
        "Museu das Ilusões Lisboa": "Museu das Ilusões Lisboa",
    }

    with patch("tools.location_resolver._fetch_nominatim_results_cached", return_value=[]):
        for query, expected_display in museum_queries.items():
            geocoded = geocode_location_name(query)
            resolved = resolve_location_query(query)

            assert geocoded is not None, query
            assert geocoded["match_source"] == "curated_gazetteer"
            assert geocoded["scope"] == "lisbon_city"
            assert expected_display in geocoded["display_name"]
            assert resolved["success"] is True
            assert resolved["nearest_metro"] or resolved["nearest_cp"]


def test_carris_geocode_uses_curated_gazetteer_for_common_museums() -> None:
    """Carris routing should receive coordinates for common museums even offline."""
    from tools.carris_api import geocode_location

    with patch("tools.location_resolver._fetch_nominatim_results_cached", return_value=[]):
        lat, lon, display = geocode_location("Museu Nacional do Azulejo")

    assert lat is not None
    assert lon is not None
    assert display == "Museu Nacional do Azulejo"


def test_transport_geocoders_resolve_station_hubs_without_poi_drift() -> None:
    """Transport geocoders should not map a hub name to a similarly named museum."""
    from tools.carris_api import geocode_location as carris_geocode
    from tools.carrismetropolitana_api import geocode_location as cm_geocode

    with patch("tools.location_resolver._fetch_nominatim_results_cached", return_value=[]):
        raw_geocode = geocode_location_name("Oriente")
        resolved = resolve_location_query("Oriente")
        carris = carris_geocode("Oriente")
        metropolitan = cm_geocode("Oriente")

    assert raw_geocode is None
    assert resolved["display_name"] == "Oriente"
    assert resolved["match_source"] == "metro_station"
    assert carris[2] == "Oriente"
    assert metropolitan is not None
    assert metropolitan["name"] == "Oriente"


def test_curated_aml_hubs_resolve_without_external_geocoder() -> None:
    """High-frequency AML hubs should have stable fallback coordinates."""
    from tools.carrismetropolitana_api import (
        geocode_location as cm_geocode,
        is_within_lisbon_city as cm_is_within_lisbon_city,
    )

    with patch("tools.location_resolver._fetch_nominatim_results_cached", return_value=[]):
        cacilhas = resolve_location_query("Cacilhas")
        cristo_rei = resolve_location_query("Cristo Rei")
        cm_cacilhas = cm_geocode("Cacilhas")

    assert cacilhas["success"] is True
    assert cacilhas["scope"] == "aml"
    assert cacilhas["match_source"] == "curated_gazetteer"
    assert cristo_rei["success"] is True
    assert cristo_rei["display_name"] == "Cristo Rei"
    assert cm_cacilhas is not None
    assert cm_cacilhas["name"] == "Cacilhas"
    assert cm_is_within_lisbon_city(38.6958, -9.1941) is True


def test_nearest_metro_for_lisbon_zoo_prefers_jardim_zoologico() -> None:
    """Lisbon Zoo should resolve to Jardim Zoológico, not the nearby Laranjeiras centroid."""
    from tools.metrolisboa_api import find_nearest_metro

    response = find_nearest_metro.invoke({"near_location_name": "Jardim Zoologico de Lisboa"})

    assert "Jardim Zoológico" in response
    assert response.index("Jardim Zoológico") < response.index("Laranjeiras")


def test_poi_name_containing_station_name_does_not_resolve_as_station() -> None:
    """POI names containing a station token should keep POI coordinates."""
    with patch("tools.location_resolver._fetch_nominatim_results_cached", return_value=[]):
        resolved = resolve_location_query("Museu do Oriente")

    assert resolved["success"] is True
    assert resolved["display_name"] == "Museu do Oriente"
    assert resolved["match_source"] == "curated_gazetteer"
    assert round(float(resolved["lat"]), 4) == 38.7031
    assert round(float(resolved["lon"]), 4) == -9.1733


def test_street_and_poi_queries_do_not_fuzzy_match_station_names() -> None:
    """Street and venue-like names must resolve as places, not nearby same-token stations."""
    with patch("tools.location_resolver._fetch_nominatim_results_cached", return_value=[]):
        zoo = resolve_location_query("Jardim Zoologico de Lisboa")
        avenue = resolve_location_query("Avenida de Roma")
        alcantara = resolve_location_query("Alcântara")

    assert zoo["match_source"] == "curated_gazetteer"
    assert zoo["display_name"] == "Jardim Zoológico de Lisboa"
    assert zoo["nearest_metro"]["name"] == "Jardim Zoológico"

    assert avenue["match_source"] == "curated_gazetteer"
    assert avenue["display_name"] == "Avenida de Roma"
    assert avenue["nearest_metro"]["name"] != "Avenida"

    assert alcantara["match_source"] == "curated_gazetteer"
    assert alcantara["display_name"] == "Alcântara"
    assert alcantara["nearest_cp"]["name"] in {"Alcantara - Mar", "Alcantara - Terra"}


def test_cp_station_scope_uses_coordinates_not_operator_scope() -> None:
    """CP stations inside Lisbon city should not be labelled as outside-city AML places."""
    with patch("tools.location_resolver._fetch_nominatim_results_cached", return_value=[]):
        belem = resolve_location_query("Belém")
        santos = resolve_location_query("Santos")

    assert belem["match_source"] == "cp_station"
    assert belem["scope"] == "lisbon_city"
    assert santos["match_source"] == "cp_station"
    assert santos["scope"] == "lisbon_city"


def test_foreign_place_aliases_resolve_to_curated_lisbon_queries() -> None:
    """Common foreign place names should resolve through curated Lisbon aliases before geocoding."""
    belem_variants = _build_query_variants("Tour de Belém")
    castle_variants = _build_query_variants("Le château de São Jorge")

    assert belem_variants[0] == "Torre de Belém, Lisboa, Portugal"
    assert castle_variants[0] == "Castelo de São Jorge, Lisboa, Portugal"


def test_nominatim_requests_are_bounded_to_portugal_and_the_aml() -> None:
    """Live geocoding should stay scoped to Portugal and the AML viewbox to reduce false positives."""

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict]:
            return []

    captured: dict[str, object] = {}
    _fetch_nominatim_results_cached.cache_clear()

    def fake_get(url, params, headers, timeout):
        captured["url"] = url
        captured["params"] = dict(params)
        captured["headers"] = dict(headers)
        captured["timeout"] = timeout
        return DummyResponse()

    with patch("tools.location_resolver.requests.get", side_effect=fake_get):
        result = _fetch_nominatim_results_cached("Rossio")

    params = captured.get("params")

    assert isinstance(params, dict)
    assert result == []
    assert params["countrycodes"] == "pt"
    assert params["bounded"] == 1
    assert params["viewbox"] == (
        f"{AML_BOUNDS['lon_min']},{AML_BOUNDS['lat_max']},"
        f"{AML_BOUNDS['lon_max']},{AML_BOUNDS['lat_min']}"
    )
