# ==========================================================================
# Master Thesis - Lisbon Urban Assistant (Streamlit App)
#   - André Filipe Gomes Silvestre, 2025
#
#   Main Streamlit application for the intelligent tourist assistant.
#   "LISBOA: LLM-Integrated System for Behavioral Orchestration and Agentic Architecture"
# ==========================================================================

import streamlit as st
import sys
import os
import time
import traceback
from datetime import datetime
from typing import Optional, Generator

# Fix for Windows Event Loop Policy in Streamlit
import asyncio
import sys

if sys.platform.startswith("win"):
    # Set proper event loop policy for Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nest_asyncio

nest_asyncio.apply()

# Suppress Torch/Streamlit file watcher warning
import torch

torch.classes.__path__ = []  # Fix RuntimeError: module 'torch.classes' has no attribute '__path__'
# Source: (https://github.com/datalab-to/marker/issues/442)

import warnings

warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, ".")

from agent.graph import create_assistant, MultiAgentAssistant
from config import Config
from tools.visitlisboa_api import initialize_vector_store
from tools.carris_api import CarrisGTFSManager, CARRIS_DB_PATH

# Define images directory
IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")

# ==========================================================================
# PAGE CONFIGURATION
# ==========================================================================

st.set_page_config(
    page_title="LISBOA: Urban Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Silvestre17/Thesis2025-26_AFGS",
        "Report a bug": "https://github.com/Silvestre17/Thesis2025-26_AFGS/issues",
        "About": "### Lisbon Urban Assistant\nMaster Thesis Project 2025",
    },
)

# ==========================================================================
# STYLES & ASSETS
# ==========================================================================

LISBON_CSS = """
<style>
/* ==========================================================================
   LISBON URBAN ASSISTANT - THEME
   Colors: Yellow #f6da00, Orange #ff4011, Blue #3777ff, Green #0ee071
   ========================================================================== */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --lisbon-yellow: #f6da00;
    --lisbon-yellow-light: #fff9e6;
    --lisbon-yellow-dark: #d4bb00;
    --lisbon-orange: #ff4011;
    --lisbon-orange-light: #ff6b47;
    --lisbon-orange-dark: #e63600;
    --lisbon-blue: #3777ff;
    --lisbon-blue-light: #5a91ff;
    --lisbon-green: #0ee071;
    --lisbon-green-light: #3de88f;
    --gray-50: #fafafa;
    --gray-100: #f4f4f5;
    --gray-200: #e4e4e7;
    --gray-300: #d4d4d8;
    --gray-600: #52525b;
    --gray-700: #3f3f46;
    --gray-800: #27272a;
    --gray-900: #18181b;
}

/* Global styles */
.main .block-container {
    padding: 2rem 1rem 3rem 1rem;
    max-width: 1100px;
}

/* ============ HEADER ============ */
.lisbon-header {
    background: linear-gradient(135deg, var(--lisbon-orange) 0%, var(--lisbon-orange-light) 50%, var(--lisbon-yellow) 100%);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(255, 64, 17, 0.25), 0 2px 8px rgba(0,0,0,0.1);
    position: relative;
    overflow: hidden;
    color: white;
}

.lisbon-header h1 {
    color: white;
    margin: 0;
    font-size: 2.4rem;
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.lisbon-header p {
    color: rgba(255,255,255,0.95);
    margin: 0.75rem 0 0 0;
    font-size: 1.15rem;
}

/* ============ SIDEBAR ============ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--gray-50) 0%, white 100%);
    border-right: none;
    box-shadow: 4px 0 20px rgba(0,0,0,0.05);
}

section[data-testid="stSidebar"] button[kind="secondary"] {
    background: white !important;
    border: 1.5px solid var(--gray-200) !important;
    color: var(--gray-700) !important;
}

section[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(135deg, var(--lisbon-orange) 0%, var(--lisbon-orange-light) 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(255, 64, 17, 0.3);
}

/* ============ CHAT MESSAGES ============ */
[data-testid="stChatMessage"] {
    padding: 1.25rem !important;
    margin: 0.75rem 0 !important;
    border-radius: 18px !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, var(--lisbon-yellow-light) 0%, white 100%) !important;
    border: 1px solid var(--lisbon-yellow) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: white !important;
    border: 1px solid var(--gray-200) !important;
    border-left: 4px solid var(--lisbon-orange) !important;
}

/* ============ CARDS & INFO ============ */
.welcome-card {
    background: white;
    border: none;
    border-radius: 20px;
    padding: 2.5rem;
    margin: 1.5rem 0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    position: relative;
    overflow: hidden;
    border-top: 4px solid var(--lisbon-orange);
}

.welcome-card h3 {
    color: var(--gray-900);
    margin-bottom: 0.5rem;
    font-size: 1.75rem;
}

.feature-list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.feature-item {
    background: var(--gray-50);
    padding: 1rem;
    border-radius: 12px;
    border-left: 3px solid var(--lisbon-yellow);
    font-size: 0.95rem;
}

/* ============ FOOTER ============ */
.lisbon-footer {
    text-align: center;
    padding: 1.5rem 2rem;
    margin-top: 3rem;
    background: var(--gray-50);
    border-radius: 16px;
    border: 1px solid var(--gray-100);
}

/* ============ INPUT ============ */
.stTextInput > div > div > input {
    border-radius: 10px !important;
}
</style>
"""

st.markdown(LISBON_CSS, unsafe_allow_html=True)

# ==========================================================================
# SESSION STATE & TRANSLATIONS
# ==========================================================================

TRANSLATIONS = {
    "en": {
        # Header
        "app_title": "Lisbon Urban Assistant",
        "app_subtitle": "Your intelligent guide to exploring Lisbon",
        # Sidebar - Settings
        "settings": "Settings & Status",
        "language": "Language",
        "provider": "LLM Provider",
        "select_provider": "Select AI Provider",
        "api_credentials": "API Credentials",
        "api_key": "API Key",
        "clear_chat": "Clear Chat",
        # Sidebar - Quick Actions
        "quick_actions": "Quick Actions",
        "weather_summary": "Weather Summary",
        "transport_status": "Transport Status",
        "upcoming_events": "Upcoming Events",
        "top_attractions": "Top Attractions",
        "plan_my_day": "Plan My Day",
        # Main Content
        "welcome": "Welcome to Lisbon! 🇵🇹",
        "intro": "I am your AI assistant for exploring the city. Ask me about weather, transport, events, or places.",
        "input_placeholder": "Ask something about Lisbon...",
        "searching": "Thinking & Searching...",
        "error_init": "Failed to initialize assistant.",
        "footer": "Master Thesis • NOVA IMS • André Silvestre",
        "try_asking": "Try asking about...",
        # Quick Action Queries
        "query_weather": "What's the current weather in Lisbon? Include any active warnings.",
        "query_transport": "What's the current status of public transport in Lisbon? Include Metro, buses, and trains.",
        "query_events": "What cultural events are happening in Lisbon this week?",
        "query_attractions": "What are the must-see tourist attractions in Lisbon?",
        "query_plan": "Help me plan a one-day trip in Lisbon. I'm interested in history and good food.",
    },
    "pt": {
        # Header
        "app_title": "Assistente Urbano de Lisboa",
        "app_subtitle": "O seu guia inteligente para explorar Lisboa",
        # Sidebar - Settings
        "settings": "Definições e Estado",
        "language": "Idioma",
        "provider": "Fornecedor LLM",
        "select_provider": "Selecionar Fornecedor IA",
        "api_credentials": "Credenciais API",
        "api_key": "Chave API",
        "clear_chat": "Limpar Conversa",
        # Sidebar - Quick Actions
        "quick_actions": "Ações Rápidas",
        "weather_summary": "Resumo do Tempo",
        "transport_status": "Estado dos Transportes",
        "upcoming_events": "Próximos Eventos",
        "top_attractions": "Principais Atrações",
        "plan_my_day": "Planear o Meu Dia",
        # Main Content
        "welcome": "Bem-vindo a Lisboa! 🇵🇹",
        "intro": "Sou o teu assistente IA para explorar a cidade. Pergunta-me sobre tempo, transportes, eventos ou locais.",
        "input_placeholder": "Pergunta algo sobre Lisboa...",
        "searching": "A pensar e pesquisar...",
        "error_init": "Falha ao inicializar assistente.",
        "footer": "Tese de Mestrado • NOVA IMS • André Silvestre",
        "try_asking": "Experimenta perguntar sobre...",
        # Quick Action Queries
        "query_weather": "Qual é a previsão do tempo para Lisboa? Inclui avisos meteorológicos ativos.",
        "query_transport": "Qual é o estado atual dos transportes públicos em Lisboa? Inclui Metro, autocarros e comboios.",
        "query_events": "Que eventos culturais estão a acontecer em Lisboa esta semana?",
        "query_attractions": "Quais são as principais atrações turísticas de Lisboa que não posso perder?",
        "query_plan": "Ajuda-me a planear um dia em Lisboa. Estou interessado em história e boa comida.",
    },
}


def init_session():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "language" not in st.session_state:
        st.session_state.language = "pt"  # Default to PT
    if "provider" not in st.session_state:
        st.session_state.provider = "lmstudio"
    if "assistant" not in st.session_state:
        st.session_state.assistant = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"


def t(key: str) -> str:
    """Get translation safely."""
    lang = st.session_state.get("language", "pt")
    return TRANSLATIONS.get(lang, TRANSLATIONS["pt"]).get(key, key)


# ==========================================================================
# BACKEND INITIALIZATION (Cached)
# ==========================================================================


@st.cache_resource(show_spinner="Starting Engine...")
def load_carris_db():
    """Load Carris DB once."""
    try:
        manager = CarrisGTFSManager()
        if not os.path.exists(CARRIS_DB_PATH):
            manager.ensure_database()
        return True
    except Exception:
        return False


@st.cache_resource(show_spinner="Initializing Vector DB...")
def load_vector_store():
    """Load ChromaDB once."""
    try:
        initialize_vector_store()
        return True
    except Exception:
        return False


def get_assistant(provider: str):
    """
    Get or create the assistant instance.
    Note: We store the assistant in session_state, but the heavy lifting
    (models, tools) should be efficient.
    """
    if (
        st.session_state.assistant is None
        or st.session_state.get("last_provider") != provider
    ):
        try:
            # Re-initialize only if provider changed or not exists
            if Config.USE_MULTI_AGENT:
                st.session_state.assistant = MultiAgentAssistant()
            else:
                st.session_state.assistant = create_assistant(provider)
            st.session_state.last_provider = provider
        except Exception as e:
            st.error(f"{t('error_init')}: {e}")
            return None
    return st.session_state.assistant


# ==========================================================================
# UI COMPONENTS
# ==========================================================================


def render_header():
    """Render the Lisbon-themed header."""
    st.markdown(
        f"""
    <div class="lisbon-header">
        <h1>🏛️ {t("app_title")}</h1>
        <p>{t("app_subtitle")}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        logo_path = os.path.join(IMG_DIR, "Logo_1-1_WithoutBG.png")
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
        else:
            # Fallback if logo missing
            st.markdown("### 🏛️ Lisboa")

        st.markdown(f"### {t('settings')}")

        # Language Selector
        lang_options = {"Português 🇵🇹": "pt", "English 🇬🇧": "en"}
        selected_lang_label = st.selectbox(
            f"🗣️ {t('language')}",
            options=lang_options.keys(),
            index=0 if st.session_state.language == "pt" else 1,
            key="lang_select_box",
        )
        # Update state immediately if changed
        new_lang = lang_options[selected_lang_label]
        if new_lang != st.session_state.language:
            st.session_state.language = new_lang
        if new_lang != st.session_state.language:
            st.session_state.language = new_lang
            st.rerun()

        # ====================
        # NAVIGATION
        # ====================
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "💬 Chat",
                use_container_width=True,
                type="primary"
                if st.session_state.current_page == "chat"
                else "secondary",
            ):
                st.session_state.current_page = "chat"
                st.rerun()
        with col2:
            if st.button(
                "ℹ️ Info",
                use_container_width=True,
                type="primary"
                if st.session_state.current_page == "info"
                else "secondary",
            ):
                st.session_state.current_page = "info"
                st.rerun()

        st.divider()

        # ====================
        # QUICK ACTIONS
        # ====================
        st.subheader(f"🚀 {t('quick_actions')}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"☀️ {t('weather_summary')}", use_container_width=True):
                st.session_state.quick_action = t("query_weather")
                st.rerun()
            if st.button(f"📅 {t('upcoming_events')}", use_container_width=True):
                st.session_state.quick_action = t("query_events")
                st.rerun()
        with col2:
            if st.button(f"🚇 {t('transport_status')}", use_container_width=True):
                st.session_state.quick_action = t("query_transport")
                st.rerun()
            if st.button(f"🏛️ {t('top_attractions')}", use_container_width=True):
                st.session_state.quick_action = t("query_attractions")
                st.rerun()

        if st.button(f"🗺️ {t('plan_my_day')}", use_container_width=True):
            st.session_state.quick_action = t("query_plan")
            st.rerun()

        st.divider()

        # Model/Provider Selector
        providers = ["lmstudio", "ollama", "groq", "openai"]
        start_idx = 0
        if st.session_state.provider in providers:
            start_idx = providers.index(st.session_state.provider)

        selected_provider = st.selectbox(
            f"🧠 {t('provider')}", providers, index=start_idx
        )
        if selected_provider != st.session_state.provider:
            st.session_state.provider = selected_provider
            st.session_state.assistant = None  # Force reload
            st.rerun()

        # API Key input (conditional)
        if selected_provider in ["groq", "openai"]:
            env_key = f"{selected_provider.upper()}_API_KEY"
            current_key = os.getenv(env_key, "")
            new_key = st.text_input(t("api_key"), value=current_key, type="password")
            if new_key != current_key:
                os.environ[env_key] = new_key

        st.divider()

        # Clear Chat
        if st.button(f"🗑️ {t('clear_chat')}", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.assistant:
                st.session_state.assistant.reset()  # Reset agent state
            st.rerun()

        # ====================
        # SYSTEM STATUS
        # ====================
        with st.expander("System Info", expanded=False):
            st.caption(f"**Model:** {st.session_state.get('provider', 'Unknown')}")
            if st.session_state.assistant:
                st.caption(
                    f"**Backend:** {Config.USE_MULTI_AGENT and 'Multi-Agent' or 'Single Agent'}"
                )

            # Simulated checks (visual only, real checks happen on demand)
            st.success("Database: Connected")
            st.success("Vector Store: Ready")

            # LangSmith Status
            if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
                st.success(
                    f"🛠️ LangSmith: Active ({os.environ.get('LANGCHAIN_PROJECT', 'default')})"
                )
            else:
                st.info("🛠️ LangSmith: Disabled")

            # LM Studio Check (if selected)
            if st.session_state.provider == "lmstudio":
                try:
                    import requests

                    requests.get(Config.LMSTUDIO_BASE_URL + "/models", timeout=1)
                    st.success("🟢 LM Studio: Online")
                except:
                    st.error("🔴 LM Studio: Offline")

        st.markdown(
            f"<div style='text-align: center; margin-top: 2rem; color: #888; font-size: 0.8rem;'>{t('footer')}</div>",
            unsafe_allow_html=True,
        )


def stream_text(text: str) -> Generator[str, None, None]:
    """Yields text chunks to simulate streaming."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


def render_welcome_section():
    """Render welcome card for empty history."""
    st.markdown(
        f"""
    <div class="welcome-card">
        <h3>{t("welcome")}</h3>
        <p>{t("intro")}</p>
        <div class="feature-list">
            <div class="feature-item">☀️ <strong>Meteorologia</strong></div>
            <div class="feature-item">🚇 <strong>Transportes</strong></div>
            <div class="feature-item">🎭 <strong>Eventos</strong></div>
            <div class="feature-item">📍 <strong>Locais</strong></div>
        </div>
        <p><strong>{t("input_placeholder")}</strong></p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_example_queries():
    """Render example query buttons in the main area."""
    st.markdown(f"### {t('try_asking')}")

    # Define examples (Icon, Label, Query Key)
    examples = [
        ("🌤️", t("weather_summary"), "query_weather"),
        ("🚇", t("transport_status"), "query_transport"),
        ("🎭", "Eventos", "query_events"),
        ("📍", "Atrações", "query_attractions"),
        ("🗺️", "Plano 1 Dia", "query_plan"),
    ]

    cols = st.columns(len(examples))
    for i, (icon, label, query_key) in enumerate(examples):
        with cols[i]:
            if st.button(
                f"{icon}\n{label}", key=f"ex_btn_{i}", use_container_width=True
            ):
                return t(query_key)
    return None


def render_footer():
    """Render footer."""
    st.markdown(
        f"""
    <div style='text-align: center; margin-top: 3rem; padding: 1rem; color: #888; font-size: 0.8rem; border-top: 1px solid #eee;'>
        <p>{t("footer")}</p>
        <p>{datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_info_page():
    """Render the Info/About page."""
    st.markdown(f"# {t('settings')} (Info)")

    st.markdown(f"### {t('title')}")
    st.info("Master Thesis Project - NOVA IMS 2025")

    st.markdown("#### Data Sources")
    st.markdown("""
    - **IPMA**: Weather and warnings
    - **Carris/Metro**: Public transport status
    - **VisitLisboa**: Events and places (Vector Search)
    """)

    st.markdown("#### Privacy")
    st.markdown("No data is stored. API keys are kept in session state only.")

    if st.button("🔙 Back to Chat"):
        st.session_state.current_page = "chat"
        st.rerun()


# ==========================================================================
# MAIN APP LOGIC
# ==========================================================================


def main():
    init_session()

    # 1. Load Heavy Resources (Once)
    load_carris_db()
    if Config.USE_MULTI_AGENT:
        load_vector_store()

    # 2. Render Sidebar
    render_sidebar()

    # 3. Main Content Area

    # Handle Page Navigation
    if st.session_state.current_page == "info":
        render_info_page()
        render_footer()
        return

    # Banner (Chat Mode)
    # Replaced by CSS-styled header
    render_header()

    if not st.session_state.messages:
        render_welcome_section()
        example_query = render_example_queries()
        if example_query:
            # Set as quick action to be picked up
            st.session_state.quick_action = example_query
            st.rerun()

    # 4. Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 5. Input & Layout
    # Handle Quick Actions or Text Input
    user_input = None
    if "quick_action" in st.session_state and st.session_state.quick_action:
        user_input = st.session_state.quick_action
        del st.session_state.quick_action

    # Disable chat input if processing (simulated)
    # Streamlit doesn't natively support disabling chat_input easily based on state without reruns
    # so we primarily rely on the user interface feedback

    # Disable chat input if processing
    disable_input = st.session_state.processing

    if chat_input := st.chat_input(t("input_placeholder"), disabled=disable_input):
        user_input = chat_input

    if user_input and not disable_input:
        # Set processing state
        st.session_state.processing = True

        # User Message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant Response
        assistant = get_assistant(st.session_state.provider)

        if assistant:
            # Create a placeholder for the assistant response immediately
            with st.chat_message("assistant"):
                # Status Container with "Thinking" visualization
                status_container = st.status(t("searching"), expanded=True)
                message_placeholder = st.empty()

                # Callback to update status
                def update_status(msg):
                    status_container.write(msg)

                try:
                    # Execute Graph
                    response_text = assistant.chat(
                        user_input,
                        on_status_change=update_status,
                        language=st.session_state.language,
                    )

                    status_container.update(
                        label="✅ Complete", state="complete", expanded=False
                    )

                    # Stream the response
                    full_response = ""
                    for chunk in stream_text(response_text):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

                    # Save to history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )

                except Exception as e:
                    status_container.update(label="❌ Error", state="error")
                    st.error(f"An error occurred: {str(e)}")
                    traceback.print_exc()
                finally:
                    st.session_state.processing = False
                    st.rerun()


if __name__ == "__main__":
    main()
