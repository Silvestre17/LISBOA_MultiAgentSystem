# ===========================================================================
# Master Thesis - Prompt Language Parity Tests
#   - André Filipe Gomes Silvestre, 20240502
#
# Focused regressions for EN/PT prompt isolation and parity across the
# prompt layer. These tests pin down past leakage bugs such as Portuguese
# content appearing inside English QA prompts.
# ===========================================================================

import pytest

from agent.prompts.planner import get_planner_prompt
from agent.prompts.qa import get_qa_prompt
from agent.prompts.researcher import get_researcher_prompt
from agent.prompts._system_prompt import get_system_prompt
from agent.prompts.supervisor import get_supervisor_prompt
from agent.prompts.transport import get_transport_prompt
from agent.prompts.weather import get_weather_prompt


def test_qa_english_prompt_does_not_leak_portuguese_hygiene_rules() -> None:
    """The English QA prompt should not contain the Portuguese hygiene bullets."""
    prompt = get_qa_prompt("en")

    assert "Higiene do output" not in prompt
    assert "Higiene de links" not in prompt
    assert "Output hygiene" in prompt
    assert "Link hygiene" in prompt


def test_qa_portuguese_prompt_contains_portuguese_hygiene_rules() -> None:
    """The Portuguese QA prompt should contain the localized hygiene bullets."""
    prompt = get_qa_prompt("pt")

    assert "Higiene do output" in prompt
    assert "Higiene de links" in prompt


@pytest.mark.parametrize(
    ("getter", "en_marker", "pt_marker", "en_absent", "pt_absent"),
    [
        (get_weather_prompt, "No active weather warnings for Lisbon", "Sem avisos meteorológicos ativos para Lisboa", "Sem avisos meteorológicos ativos para Lisboa", "No active weather warnings for Lisbon"),
        (get_transport_prompt, "Estimated total time", "Tempo total estimado", "Tempo total estimado", "Estimated total time"),
        (get_researcher_prompt, "**Opening hours**", "**Horário**", "**Horário**", "**Opening hours**"),
        (get_planner_prompt, "Itinerary for [Date]", "Itinerário para [Data]", "Itinerário para [Data]", "Itinerary for [Date]"),
    ],
)
def test_worker_prompts_select_localized_templates(
    getter,
    en_marker: str,
    pt_marker: str,
    en_absent: str,
    pt_absent: str,
) -> None:
    """Worker prompt getters should return localized template blocks per requested language."""
    prompt_en = getter(language="en")
    prompt_pt = getter(language="pt")

    assert en_marker in prompt_en
    assert pt_marker in prompt_pt
    assert en_absent not in prompt_en
    assert pt_absent not in prompt_pt


def test_researcher_prompt_removed_old_meta_commentary() -> None:
    """The researcher prompt should no longer include developer-style template commentary."""
    prompt_en = get_researcher_prompt(language="en")
    prompt_pt = get_researcher_prompt(language="pt")

    assert "ADAPT TO DETECTED LANGUAGE" not in prompt_en
    assert "ADAPT TO DETECTED LANGUAGE" not in prompt_pt
    assert "Portuguese example" not in prompt_en


def test_supervisor_portuguese_prompt_mentions_third_language_fallback() -> None:
    """The Portuguese supervisor prompt should mirror the English non-PT/non-EN fallback rule."""
    prompt_pt = get_supervisor_prompt("pt")

    assert "noutra língua" in prompt_pt
    assert "responde em Inglês" in prompt_pt


def test_main_system_prompts_do_not_use_opposite_language_meta_examples() -> None:
    """The main EN/PT system prompts should keep their meta-example wording language-consistent."""
    prompt_en = get_system_prompt(language="en")
    prompt_pt = get_system_prompt(language="pt")

    assert "Checklist de Completude" not in prompt_en
    assert "Introdução" not in prompt_en
    assert "Completeness Checklist" in prompt_en

    assert "User constraints" not in prompt_pt
    assert "Overview" not in prompt_pt
    assert "Restrições do utilizador" in prompt_pt
