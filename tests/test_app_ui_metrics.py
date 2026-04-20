# ===========================================================================
# Master Thesis - App UI Metric Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Focused regressions for sidebar metrics in the Streamlit app.
# ===========================================================================

from unittest.mock import MagicMock, patch

from app import count_user_interactions, render_assistant_markdown, select_new_request


def test_count_user_interactions_counts_only_user_turns() -> None:
    """Sidebar interaction metrics should count only user turns, not assistant replies."""
    messages = [
        {"role": "user", "content": "Pergunta 1"},
        {"role": "assistant", "content": "Resposta 1"},
        {"role": "user", "content": "Pergunta 2"},
        {"role": "assistant", "content": "Resposta 2"},
    ]

    assert count_user_interactions(messages) == 2


def test_count_user_interactions_ignores_nonstandard_entries() -> None:
    """Malformed or non-dict items must not inflate the sidebar interaction metric."""
    messages = [
        {"role": "user", "content": "Olá"},
        "assistant",
        {"role": "assistant", "content": "Resposta"},
        {"content": "sem role"},
    ]

    assert count_user_interactions(messages) == 1


def test_select_new_request_blocks_duplicate_capture_when_pending_exists() -> None:
    """A rerun with a pending request must not re-queue the same quick action again."""
    selected = select_new_request(
        sidebar_request="weather",
        welcome_request=None,
        chat_request=None,
        pending_request="weather",
    )

    assert selected is None


def test_select_new_request_keeps_chat_priority_without_pending_request() -> None:
    """Fresh chat input should still win over welcome or sidebar suggestions when nothing is pending."""
    selected = select_new_request(
        sidebar_request="sidebar weather",
        welcome_request="welcome itinerary",
        chat_request="typed question",
        pending_request=None,
    )

    assert selected == "typed question"


def test_render_assistant_markdown_rerenders_original_full_text_after_streaming() -> None:
    """The final Streamlit render must use the canonical full markdown, not the progressive chunk accumulation."""
    first_placeholder = MagicMock()
    final_placeholder = MagicMock()

    with patch("app.st.empty", side_effect=[first_placeholder, final_placeholder]), patch(
        "app.handle_chat_stream",
        return_value=iter(["### 🎭 Evento\n", "- 📍 **Morada:** Lisboa", " - 📅 **Data/Hora:** Hoje\n"]),
    ):
        final_text = render_assistant_markdown(
            "### 🎭 Evento\n- 📍 **Morada:** Lisboa\n- 📅 **Data/Hora:** Hoje\n"
        )

    first_placeholder.markdown.assert_any_call("### 🎭 Evento\n")
    first_placeholder.empty.assert_called_once()
    final_placeholder.markdown.assert_called_once_with(
        "### 🎭 Evento\n- 📍 **Morada:** Lisboa\n- 📅 **Data/Hora:** Hoje\n"
    )
    assert final_text == "### 🎭 Evento\n- 📍 **Morada:** Lisboa\n- 📅 **Data/Hora:** Hoje\n"
