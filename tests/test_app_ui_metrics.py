# ===========================================================================
# Master Thesis - App UI Metric Tests
#   - André Filipe Gomes Silvestre, 20240502
#
#   Focused regressions for sidebar metrics in the Streamlit app.
# ===========================================================================

from unittest.mock import MagicMock, patch

from app import (
    count_user_interactions,
    normalize_streamlit_chat_markdown,
    render_assistant_markdown,
    request_capture_locked,
    runtime_auto_initialize_enabled,
    runtime_settings_panel_visible,
    select_new_request,
    should_attempt_startup_auto_initialization,
)


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


def test_request_capture_locked_only_when_pending_request_exists() -> None:
    """Quick actions and chat input should lock only while one request is already pending."""
    assert request_capture_locked(None) is False
    assert request_capture_locked("") is False
    assert request_capture_locked(None, request_running=True) is True
    assert request_capture_locked("Estado dos Transportes") is True


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


def test_streamlit_markdown_normalizer_prevents_orphan_indented_code_blocks() -> None:
    """Indented LISBOA card bullets need a parent list item for Streamlit Markdown."""
    raw = (
        "### 📍 **Suggested route**\n\n"
        "**🏷️ Water Museum**\n"
        "    - 📝 **Description:** Historic reservoir.\n"
        "    - 📍 **Address:** Lisboa\n\n"
        "### 🚇 **How to move**\n"
        "    - 🚇 **Metro:** Saldanha to Avenida\n"
    )

    normalized = normalize_streamlit_chat_markdown(raw)

    assert "- **🏷️ Water Museum**" in normalized
    assert "    - 📝 **Description:** Historic reservoir." in normalized
    assert "\n- 🚇 **Metro:** Saldanha to Avenida" in normalized


def test_runtime_auto_initialize_enabled_only_in_locked_production_mode() -> None:
    """Startup auto-initialization should only activate when provider and credential editing are both disabled."""
    with patch("app.Config.ENABLE_PROVIDER_SELECTOR", False), patch(
        "app.Config.ENABLE_PROVIDER_CREDENTIAL_INPUTS", False
    ):
        assert runtime_auto_initialize_enabled() is True

    with patch("app.Config.ENABLE_PROVIDER_SELECTOR", True), patch(
        "app.Config.ENABLE_PROVIDER_CREDENTIAL_INPUTS", False
    ):
        assert runtime_auto_initialize_enabled() is False

    with patch("app.Config.ENABLE_PROVIDER_SELECTOR", False), patch(
        "app.Config.ENABLE_PROVIDER_CREDENTIAL_INPUTS", True
    ):
        assert runtime_auto_initialize_enabled() is False


def test_runtime_settings_panel_visible_when_credential_inputs_are_enabled() -> None:
    """The settings panel must stay visible when credentials are editable, even if provider selection is locked."""
    with patch("app.Config.ENABLE_PROVIDER_SELECTOR", False), patch(
        "app.Config.ENABLE_PROVIDER_CREDENTIAL_INPUTS", True
    ):
        assert runtime_settings_panel_visible() is True


def test_runtime_settings_panel_visible_when_provider_selector_is_enabled() -> None:
    """The settings panel must stay visible when provider selection is editable."""
    with patch("app.Config.ENABLE_PROVIDER_SELECTOR", True), patch(
        "app.Config.ENABLE_PROVIDER_CREDENTIAL_INPUTS", False
    ):
        assert runtime_settings_panel_visible() is True


def test_should_attempt_startup_auto_initialization_stops_after_same_provider_failure() -> None:
    """The app should not retry automatic startup initialization endlessly after a failed attempt for the same provider."""
    with patch("app.runtime_auto_initialize_enabled", return_value=True):
        assert (
            should_attempt_startup_auto_initialization(
                initialized=False,
                current_provider="azure",
                selected_provider="azure",
                credentials_ready=True,
                attempted_provider="azure",
                last_error="probe failed",
            )
            is False
        )


def test_should_attempt_startup_auto_initialization_runs_for_fresh_locked_session() -> None:
    """A fresh locked production session with valid credentials should auto-initialize immediately."""
    with patch("app.runtime_auto_initialize_enabled", return_value=True):
        assert (
            should_attempt_startup_auto_initialization(
                initialized=False,
                current_provider="azure",
                selected_provider="azure",
                credentials_ready=True,
                attempted_provider=None,
                last_error=None,
            )
            is True
        )
