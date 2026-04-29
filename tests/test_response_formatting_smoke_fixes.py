# ==========================================================================
# Master Thesis - Response Formatting Smoke Fix Tests
#   - André Filipe Gomes Silvestre, 20240502
#
# Regression tests for focused visual cleanup fixes added after the April 2026
# smoke-test review. These tests avoid the large guardrails test module and
# cover final rendering helpers used by the Streamlit response path.
#
# Features:
#   - Weather detail indentation and tip spacing validation
#   - Transport ambiguity deduplication and GTFS wording cleanup
#   - Portuguese phone linkification for 00351-prefixed phone numbers
# Usage:
#   > python -m pytest tests/test_response_formatting_smoke_fixes.py -q
# Notes:
#   - Tests are pure formatter checks and do not call live APIs or LLMs.
# ==========================================================================

from __future__ import annotations

from agent.agents.researcher_agent import ResearcherAgent
from agent.utils.response_formatter import final_visual_pass


def test_final_visual_pass_indents_weather_conditions_and_tips() -> None:
    """Weather day details should stay nested and tips should start a new block."""
    raw = """### 🌤️ Previsão Meteorológica
- **📅 Quarta-feira, 29 de abril**
    - 🌡️ **Temperatura**: 14.0°C a 19.4°C
☁️ **Condições**: Aguaceiros/fraca chuva
    - 💧 **Chuva**: 98% - fraca
💡 **Dicas Práticas**
Leva guarda-chuva.

📌 **Fonte:** Dados do [*IPMA*](https://www.ipma.pt) | **Atualizado:** 17:54"""

    output = final_visual_pass(raw)

    assert "    - ☁️ **Condições**: Aguaceiros/fraca chuva" in output
    assert "\n\n💡 **Dicas Práticas**\n" in output


def test_final_visual_pass_deduplicates_marques_ambiguity_block() -> None:
    """Duplicated Marquês ambiguity options should be shown only in the preamble."""
    raw = """⚠️ **Ambiguidade em 'Marquês':** posso estar a interpretar uma destas opções:

A) 🚇 **Estação Marquês de Pombal** — ligação Metro Azul e Amarela.
B) 📍 **Praça/Rotunda do Marquês de Pombal** — superfície e avenida envolvente.

### 🚇 Mobilidade em Lisboa

A) 🚇 **Estação Marquês de Pombal** — ligação Metro Azul e Amarela.
B) 📍 **Praça/Rotunda do Marquês de Pombal** — superfície e avenida envolvente.

**Trajeto:** Marquês de Pombal → Belém
📡 **Tempo real:** Carris GTFS-RT ativo.

📌 **Fonte:** [*Carris*](https://www.carris.pt) | **Atualizado:** 17:43"""

    output = final_visual_pass(raw)

    assert output.count("Estação Marquês de Pombal") == 1
    assert "- A) 🚇 **Estação Marquês de Pombal**" in output
    assert "- B) 📍 **Praça/Rotunda do Marquês de Pombal**" in output
    assert "Carris em tempo real ativo" in output


def test_final_visual_pass_nests_transport_option_fields() -> None:
    """Transport option fields should render as nested bullets in Streamlit."""
    raw = """**🚌 Autocarros**

- 🚌 **Linha 723** — para Algés
   🕐 **Próximas partidas:** 19:45, 19:59
   ⏱️ **Tempo estimado de viagem:** ~31 min"""

    output = final_visual_pass(raw)

    assert "- 🚌 **Linha 723** — para Algés" in output
    assert "  - 🕐 **Próximas partidas:** 19:45, 19:59" in output
    assert "  - ⏱️ **Tempo estimado de viagem:** ~31 min" in output


def test_final_visual_pass_separates_transport_comparison_fields() -> None:
    """Transport comparison details should not collapse into one Streamlit paragraph."""
    raw = """### 🚇 Mobilidade em Lisboa

    **Comparação:** Entrecampos → Sete Rios

#### 🚇 Metro de Lisboa

⏱️ **Tempo estimado:** 21 min
✅ **Estado:** linhas sem perturbação reportada
🧭 **Trajeto Metro:**
🟡 Linha Amarela - direção Rato
🔄 Transferência em Marquês de Pombal
🔵 Linha Azul - direção Reboleira
🎯 Saia na estação Jardim Zoológico
ℹ️ **Sete Rios no Metro:** a estação que serve Sete Rios chama-se Jardim Zoológico.

#### 🚆 Comboio

⏱️ **Tempo estimado:** 2 min 📍 **Percurso:** embarca em Entrecampos e sai em Sete Rios
🚆 **Ligação:** direta nas partidas mostradas"""

    output = final_visual_pass(raw)

    assert "⏱️ **Tempo estimado:** 21 min\n\n✅ **Estado:**" in output
    assert "🧭 **Trajeto Metro:**\n\n🟡 Linha Amarela" in output
    assert "🎯 Saia na estação Jardim Zoológico\n\nℹ️ **Sete Rios no Metro:**" in output
    assert "ℹ️ **Sete Rios no Metro:** a estação que serve Sete Rios chama-se Jardim Zoológico.\n\n---\n\n### 🚆 Comboio" in output
    assert "- ⏱️ **Tempo estimado:** 2 min" in output
    assert "- 📍 **Percurso:** embarca em Entrecampos e sai em Sete Rios" in output
    assert "- 🚆 **Ligação:** direta nas partidas mostradas" in output


def test_final_visual_pass_separates_plain_transport_labels() -> None:
    """Transport field splitting should work after QA removes label bolding."""
    raw = "⏱️ Tempo estimado: 2 min 📍 Percurso: embarca em Entrecampos e sai em Sete Rios"

    output = final_visual_pass(raw)

    assert "⏱️ Tempo estimado: 2 min\n\n📍 Percurso:" in output


def test_final_visual_pass_separates_newline_time_and_route_fields() -> None:
    """A single newline between transport time and route should become a paragraph break."""
    raw = "⏱️ **Tempo estimado:** 2 min\n📍 **Percurso:** embarca em Entrecampos"

    output = final_visual_pass(raw)

    assert "⏱️ **Tempo estimado:** 2 min\n\n📍 **Percurso:**" in output


def test_final_visual_pass_bulletizes_plain_train_comparison_block() -> None:
    """Plain-label Metro vs train comparisons should render the train details as bullets."""
    raw = """### 🚇 Mobilidade em Lisboa

Comparação: Entrecampos → Sete Rios

🚇 Metro de Lisboa

⏱️ Tempo estimado: 21 min

ℹ️ Sete Rios no Metro: a estação que serve Sete Rios chama-se Jardim Zoológico.

🚆 Comboio

⏱️ Tempo estimado: 2 min
📍 Percurso: embarca em Entrecampos e sai em Sete Rios
🚆 Ligação: direta nas partidas mostradas
📡 Tempo real CP: sem dados em tempo real no feed usado
🚆 Linhas: Linha de Sintra, Linha da Azambuja
🕐 Próximas saídas mostradas: 22:17, 22:22, 22:47 ✅ Conclusão
Mais rápido: Comboio"""

    output = final_visual_pass(raw)

    assert "ℹ️ Sete Rios no Metro: a estação que serve Sete Rios chama-se Jardim Zoológico.\n\n---\n\n🚆 Comboio" in output
    assert "- ⏱️ Tempo estimado: 2 min" in output
    assert "- 📍 Percurso: embarca em Entrecampos e sai em Sete Rios" in output
    assert "- 🕐 Próximas saídas mostradas: 22:17, 22:22, 22:47" in output
    assert "- 🕐 Próximas saídas mostradas: 22:17, 22:22, 22:47\n\n---\n\n**✅ Conclusão**" in output
    assert "22:47 ✅ Conclusão" not in output


def test_final_visual_pass_linkifies_00351_phone_numbers() -> None:
    """Raw 00351 phone numbers should become readable +351 tel links."""
    raw = "📞 **Phone:** 00351218700365"

    output = final_visual_pass(raw)

    assert "[+351 218 700 365](tel:+351218700365)" in output
    assert "00351218700365" not in output


def test_researcher_extracts_nearest_service_location_with_suffix() -> None:
    """Nearest-service English queries should preserve the location before a trailing clause."""
    location = ResearcherAgent._extract_near_location_name(
        "Where is the nearest pharmacy to Parque das Nações that should still be useful this evening?"
    )

    assert location == "Parque das Nações"


def test_final_visual_pass_splits_municipal_service_fields() -> None:
    """Municipal service results should not render name, address, and distance on one line."""
    raw = "💊 Farmácia Dalva 📍 Morada: Avenida Duque d'Ávila, 125 📏 Distância: 0.07 km 🗺️ Coordenadas: 38.735010, -9.145924"

    output = final_visual_pass(raw)

    assert "- 💊 **Farmácia Dalva**" in output
    assert "    - 📍 Morada: Avenida Duque d'Ávila, 125" in output
    assert "    - 📏 Distância: 0.07 km" in output
    assert "    - 🗺️ Coordenadas: [38.735010, -9.145924]" in output
    assert "https://www.google.com/maps/search/?api=1&query=38.735010%2C-9.145924" in output


def test_final_visual_pass_converts_multiline_municipal_fields_to_nested_bullets() -> None:
    """Already split municipal fields should still become nested bullets."""
    raw = """- 💊 **Farmácia Dalva**
   📍 **Morada:** Avenida Duque d'Ávila, 125
   📏 **Distância:** 0.07 km"""

    output = final_visual_pass(raw)

    assert "- 💊 **Farmácia Dalva**\n    - 📍 **Morada:** [Avenida Duque d'Ávila, 125]" in output
    assert "https://www.google.com/maps/search/?api=1&query=Avenida+Duque+d%27%C3%81vila%2C+125" in output
    assert "    - 📏 **Distância:** 0.07 km" in output


def test_final_visual_pass_converts_warning_bullets_to_blocks() -> None:
    """Warning bullets should become standalone paragraphs before the source."""
    raw = """- 🚌 **Linha 732** — Hosp. Sta. Maria - Caselas

- ⚠️ Os números das linhas devem ser confirmados em carris.pt.

📌 **Fonte:** [*Carris*](https://www.carris.pt)"""

    output = final_visual_pass(raw)

    assert "- ⚠️" not in output
    assert "\n\n⚠️ Os números das linhas devem ser confirmados em carris.pt.\n\n📌" in output
