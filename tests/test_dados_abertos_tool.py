# ==========================================================================
# Master Thesis - Lisboa Aberta Tool Tests
#   - André Filipe Gomes Silvestre, 20240502
#
# Focused regressions for the Lisboa Aberta nearby-service dataset fallback
# path used by municipal-service answers.
# ==========================================================================

from __future__ import annotations

import pandas as pd

import tools.dados_abertos as dados_abertos


def _point_feature(name: str, address: str) -> dict:
    """Build a minimal GeoJSON point feature for nearby-service tests."""
    return {
        "type": "Feature",
        "properties": {"name": name, "address": address},
        "geometry": {"type": "Point", "coordinates": [-9.14, 38.72]},
    }


def test_find_nearby_services_skips_failing_datasets_and_uses_next_candidate(monkeypatch) -> None:
    """A broken first dataset should not abort nearby-service answers when later matches work."""
    matches = pd.DataFrame(
        [
            {"title": "Broken Hospitals", "description": "Hospitais", "stable_url": "https://broken.example"},
            {"title": "Hospitais Públicos", "description": "Hospitais", "stable_url": "https://good.example"},
        ]
    )

    monkeypatch.setattr(dados_abertos, "search_datasets", lambda service_type: matches)
    monkeypatch.setattr(dados_abertos, "get_datasets_for_category", lambda category: pd.DataFrame())

    def fake_fetch(url: str):
        if url == "https://broken.example":
            return None
        return {"features": [_point_feature("Hospital de Lisboa", "Rua A")]}

    monkeypatch.setattr(dados_abertos, "fetch_geojson_with_retry", fake_fetch)

    result = str(dados_abertos.find_nearby_services.invoke({"service_type": "hospitais", "max_results": 1}))

    assert "Hospital de Lisboa" in result
    assert "Broken Hospitals" not in result


def test_find_nearby_services_prefers_service_matches_over_broad_category_noise(monkeypatch) -> None:
    """Category filtering should fall back to service-specific dataset search, not arbitrary category datasets."""
    category_noise = pd.DataFrame(
        [{"title": "Alvarás de Obras", "description": "Obras", "stable_url": "https://noise.example"}]
    )
    service_matches = pd.DataFrame(
        [{"title": "Hospitais Públicos", "description": "Hospitais", "stable_url": "https://good.example"}]
    )

    monkeypatch.setattr(dados_abertos, "get_datasets_for_category", lambda category: category_noise)
    monkeypatch.setattr(dados_abertos, "search_datasets", lambda service_type: service_matches)
    monkeypatch.setattr(
        dados_abertos,
        "fetch_geojson_with_retry",
        lambda url: {"features": [_point_feature("Hospital de São José", "Rua B")]},
    )

    result = str(
        dados_abertos.find_nearby_services.invoke(
            {"service_type": "hospitais", "category": "saúde", "max_results": 1}
        )
    )

    assert "Hospital de São José" in result
    assert "Alvarás de Obras" not in result


def test_fetch_geojson_caches_unavailable_4xx_dataset_urls(monkeypatch) -> None:
    """A 4xx Lisboa Aberta dataset URL should be marked unavailable and not retried."""
    dados_abertos._UNAVAILABLE_DATASET_URLS.clear()
    calls = {"count": 0}

    class FakeResponse:
        status_code = 400

        def raise_for_status(self) -> None:
            raise AssertionError("raise_for_status should not be called for cached 4xx handling")

    def fake_get(url: str, timeout: int):
        calls["count"] += 1
        return FakeResponse()

    monkeypatch.setattr(dados_abertos.requests, "get", fake_get)

    assert dados_abertos.fetch_geojson_with_retry("https://services.arcgis.com/broken") is None
    assert dados_abertos.fetch_geojson_with_retry("https://services.arcgis.com/broken") is None
    assert calls["count"] == 1
    assert dados_abertos._UNAVAILABLE_DATASET_URLS["https://services.arcgis.com/broken"] == "HTTP 400"
