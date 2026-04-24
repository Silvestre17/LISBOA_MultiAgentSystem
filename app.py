# ==========================================================================
# LISBOA - Lisbon Itinerary System Based On AI
# ==========================================================================

import logging
import warnings

warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Suppress noisy LangSmith rate-limit/retry warnings that flood the terminal
# These are non-critical as LangSmith is optional tracing infrastructure
for _ls_logger_name in ("langsmith.client", "langsmith.utils", "langsmith"):
    logging.getLogger(_ls_logger_name).setLevel(logging.ERROR)

import base64
import html
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# WORKAROUND: Fix Streamlit file watcher crash with PyTorch
try:
    import torch

    class _StreamlitTorchPath:
        _path = []

    torch.classes.__path__ = _StreamlitTorchPath()
except ImportError:
    pass
except Exception:
    pass

sys.path.insert(0, ".")

# from agent.utils.langsmith_tracing import (
#     get_langsmith_display_state,
#     get_langsmith_project_name,
# )
from agent.utils.startup_resources import (
    pre_warm_transport_networks as _pre_warm_transport_networks_impl,
    pre_warm_vector_store as _pre_warm_vector_store_impl,
    prepare_transport_database as _prepare_transport_database_impl,
    run_startup_preload as _run_startup_preload_impl,
)
from config import Config

# ==========================================================================
# TRANSLATIONS / INTERNATIONALIZATION
# ==========================================================================

TRANSLATIONS = {
    "en": {
        "app_title": "LISBOA",
        "app_subtitle": "Your Intelligent Urban & Tourism Assistant",
        "settings": "System Settings",
        "language": "Language",
        "llm_provider": "AI Provider",
        "select_provider": "Select Engine",
        "api_credentials": "Authentication",
        "api_key": "API Key",
        "api_key_placeholder": "Enter key...",
        "local_url": "Server URL",
        "local_url_placeholder": "http://localhost:1234/v1",
        "model_name": "Model",
        "model_name_placeholder": "e.g., llama3.2",
        "save_credentials": "Connect System",
        "assistant_ready": "System Online",
        "initialization_failed": "Connection Failed",
        "quick_actions": "Quick Actions",
        "weather_summary": "Weather Report",
        "transport_status": "Transport Status",
        "upcoming_events": "Discover Events",
        "top_attractions": "Top Attractions",
        "plan_my_day": "Plan Itinerary",
        "session_info": "Session Status",
        "messages": "Interactions",
        "status": "Network",
        "clear_conversation": "New Session",
        "about": "Information",
        "tracing": "Tracing",
        "tracing_active": "LangSmith Active",
        "tracing_disabled": "LangSmith Disabled",
        "tracing_auto_disabled": "LangSmith Auto-disabled",
        "tracing_auto_disabled_invalid_credentials": "LangSmith Auto-disabled (invalid credentials)",
        "tracing_auto_disabled_invalid_configuration": "LangSmith Auto-disabled (invalid configuration)",
        "tracing_reason": "Reason",
        "project": "Project",
        "welcome_title": "Welcome to Lisbon!",
        "welcome_intro": "I am LISBOA, your intelligent assistant for navigating and exploring the Lisbon Metropolitan Area. How can I help today?",
        "weather_desc": "**Weather** - Live forecasts & alerts",
        "transport_desc": "**Mobility** - Real-time transit data",
        "events_desc": "**Culture** - Concerts, exhibitions & events",
        "places_desc": "**Discovery** - POIs & essential services",
        "planning_desc": "**Itineraries** - Context-aware route planning",
        "ask_anything": "What would you like to know about the city?",
        "try_asking": "Popular questions:",
        "chat_placeholder": "Ask your question here...",
        "ex_weather": "Weather",
        "ex_metro": "Subway",
        "ex_events": "Events",
        "ex_services": "Services",
        "ex_food": "Dining",
        "ex_planning": "Itinerary",
        "query_weather": "What is the detailed weather forecast for Lisbon today? Are there any active warnings?",
        "query_transport": "Give me a real-time status update on Lisbon's Metro, buses, and trains.",
        "query_events": "I want to explore the culture. What major events are happening this week?",
        "query_attractions": "List the highly recommended attractions for a tourist visiting Lisbon.",
        "query_plan": "Create an optimized 1-day itinerary combining historical sights and traditional cuisine.",
        "ex_query_weather": "What's the weather forecast for the next 3 days in Lisbon?",
        "ex_query_metro": "Are there any delays on the Lisbon metro?",
        "ex_query_events": "Find live music events this weekend.",
        "ex_query_services": "Where is the nearest 24h pharmacy?",
        "ex_query_food": "Find traditional Portuguese cuisine in Alfama.",
        "ex_query_planning": "Plan a 2-day walking tour for architecture lovers.",
        "error_generic": "Service temporarily unavailable. Please try again later.",
        "history_window_notice": "Showing the last {count} messages to keep the interface responsive. The full conversation is still kept for the assistant.",
        "thinking": "Analyzing live city data...",
        "footer_version": "LISBOA | AI Assistant",
        "footer_made": "André Filipe Gomes Silvestre • NOVA IMS",
        "info_title": "About LISBOA",
        "info_subtitle": "Discover the Multi-Agent Urban System",
        "info_intro": "Explore how the system is built, the integrated networks, and how it protects your privacy while delivering top-tier urban assistance.",
        "info_f1_title": "City Exploration",
        "info_f1_desc": "Historical sites and modern attractions powered by RAG.",
        "info_f2_title": "Mobility",
        "info_f2_desc": "Live Metro, Carris, and Train updates across five networks.",
        "info_f3_title": "Weather",
        "info_f3_desc": "Integrated IPMA forecast and warnings.",
        "info_f4_title": "Essential Services",
        "info_f4_desc": "Pharmacies, hospitals, and real-time open data services.",
        "info_architecture_title": "System Architecture",
        "info_architecture_desc": "Advanced multi-agent network orchestrated by LangGraph.",
        "back_to_chat": "Back to Chat",
        "feat_atmosfera": "🌤️ Atmosphere",
        "feat_mobilidade": "🚇 Mobility",
        "feat_cultura": "🎭 Culture",
        "feat_mapa": "📍 Places",
        "feat_roteiros": "🗺️ Itineraries",
        "info_objective": "System Capabilities",
        "info_objective_text": "LISBOA (Lisbon Itinerary System Based On AI) is a state-of-the-art intelligent platform serving both tourists and residents. It seamlessly integrates real-time mobility APIs, detailed meteorological forecasts, and a rich, dynamically updated database of cultural events and local landmarks to provide highly personalized, context-aware assistance.",
        "info_data_sources": "Integrated Networks",
        "info_data_sources_text": """- **IPMA API** - Live meteorological updates
- **Metro de Lisboa** - Real-time subway status
- **Carris & Carris Metropolitana** - Surface transport tracking
- **CP (Comboios de Portugal)** - Railway networks
- **Lisboa Aberta** - Essential local services
- **VisitLisboa** - Tourism and cultural repositories""",
        "info_privacy": "Privacy First",
        "info_privacy_text": "- Your API credentials are stored locally in your browser session only\n- No conversation data is stored permanently on any server\n- Geolocation data is strictly processed per query",
        "info_how_to_use": "How to Use",
        "info_how_to_use_text": """1. **Select your AI Provider** - Choose from OpenAI, Azure, or LM Studio
 2. **Enter your credentials** - Provide the required API key or server URL
 3. **Ask questions** - Type your questions in natural language
 4. **Use Quick Actions** - Click sidebar buttons for common queries""",
        "info_author": "Author",
        "info_author_text": """**André Filipe Gomes Silvestre**
Master's Student in Data Science and Advanced Analytics
NOVA IMS - Universidade NOVA de Lisboa
2025/2026""",
    },
    "pt": {
        "app_title": "LISBOA",
        "app_subtitle": "O seu Assistente Urbano Inteligente",
        "settings": "Configurações",
        "language": "Idioma / Language",
        "llm_provider": "Motor de IA",
        "select_provider": "Selecionar Motor",
        "api_credentials": "Autenticação",
        "api_key": "Chave API",
        "api_key_placeholder": "Insira a chave...",
        "local_url": "URL do Servidor",
        "local_url_placeholder": "http://localhost:1234/v1",
        "model_name": "Modelo",
        "model_name_placeholder": "ex: llama3.2",
        "save_credentials": "Ligar Sistema",
        "assistant_ready": "Sistema Online",
        "initialization_failed": "Falha na Ligação",
        "quick_actions": "Ações Rápidas",
        "weather_summary": "Boletim Meteorológico",
        "transport_status": "Estado dos Transportes",
        "upcoming_events": "Descobrir Eventos",
        "top_attractions": "Principais Atrações",
        "plan_my_day": "Criar Itinerário",
        "session_info": "Estado da Sessão",
        "messages": "Interações",
        "status": "Rede",
        "clear_conversation": "Nova Sessão",
        "about": "Informações",
        "tracing": "Rastreamento",
        "tracing_active": "LangSmith Ativo",
        "tracing_disabled": "LangSmith Desativado",
        "tracing_auto_disabled": "LangSmith Desativado Automaticamente",
        "tracing_auto_disabled_invalid_credentials": "LangSmith Desativado Automaticamente (credenciais inválidas)",
        "tracing_auto_disabled_invalid_configuration": "LangSmith Desativado Automaticamente (configuração inválida)",
        "tracing_reason": "Motivo",
        "project": "Projeto",
        "welcome_title": "Bem-vindo a Lisboa!",
        "welcome_intro": "Sou o LISBOA, o seu assistente inteligente para navegar e explorar a Área Metropolitana de Lisboa. Como posso ajudar hoje?",
        "weather_desc": "**Meteorologia** - Previsões e alertas em tempo real",
        "transport_desc": "**Mobilidade** - Dados de trânsito atualizados",
        "events_desc": "**Cultura** - Concertos, exposições e eventos",
        "places_desc": "**Descoberta** - Pontos de interesse e serviços",
        "planning_desc": "**Itinerários** - Planeamento contextualizado",
        "ask_anything": "O que gostaria de saber sobre a cidade?",
        "try_asking": "Perguntas frequentes:",
        "chat_placeholder": "Escreva a sua pergunta...",
        "ex_weather": "Tempo",
        "ex_metro": "Metro",
        "ex_events": "Eventos",
        "ex_services": "Serviços",
        "ex_food": "Gastronomia",
        "ex_planning": "Itinerário",
        "query_weather": "Qual é a previsão detalhada para Lisboa hoje? Existem avisos ativos?",
        "query_transport": "Dá-me o ponto de situação do Metro, autocarros e comboios em Lisboa.",
        "query_events": "Quero explorar a cultura local. Que grandes eventos temos esta semana?",
        "query_attractions": "Lista as atrações imperdíveis para quem visita Lisboa pela primeira vez.",
        "query_plan": "Cria um roteiro otimizado de 1 dia combinando monumentos históricos e comida tradicional.",
        "ex_query_weather": "Qual é a previsão do tempo para os próximos 3 dias?",
        "ex_query_metro": "Existem perturbações nas linhas do metro de Lisboa?",
        "ex_query_events": "Encontra eventos de música ao vivo para este fim de semana.",
        "ex_query_services": "Onde fica a farmácia de serviço mais próxima do Rossio?",
        "ex_query_food": "Onde posso comer pratos tradicionais em Alfama?",
        "ex_query_planning": "Planeia 2 dias a pé para amantes de arquitetura.",
        "error_generic": "Serviço temporariamente indisponível. Tente novamente mais tarde.",
        "history_window_notice": "A mostrar apenas as últimas {count} mensagens para manter a interface fluida. A conversa completa continua disponível para o assistente.",
        "thinking": "A processar dados urbanos...",
        "footer_version": "LISBOA | Assistente IA",
        "footer_made": "André Filipe Gomes Silvestre • NOVA IMS",
        "info_title": "Sobre o LISBOA",
        "info_subtitle": "Descubra o Sistema Urbano Multi-Agente",
        "info_intro": "Explore como o sistema é construído, as redes integradas e como protege a sua privacidade enquanto fornece assistência urbana de excelência.",
        "info_f1_title": "Exploração da Cidade",
        "info_f1_desc": "Locais históricos e atrações modernas.",
        "info_f2_title": "Mobilidade",
        "info_f2_desc": "Atualizações em tempo real do Metro, Carris e Comboios.",
        "info_f3_title": "Meteorologia",
        "info_f3_desc": "Previsões e avisos meteorológicos integrados do IPMA.",
        "info_f4_title": "Serviços Essenciais",
        "info_f4_desc": "Farmácias, hospitais e serviços em tempo real.",
        "info_architecture_title": "Arquitetura do Sistema",
        "info_architecture_desc": "Rede multi-agente avançada orquestrada por LangGraph.",
        "back_to_chat": "Voltar ao Chat",
        "feat_atmosfera": "🌤️ Atmosfera",
        "feat_mobilidade": "🚇 Mobilidade",
        "feat_cultura": "🎭 Cultura",
        "feat_mapa": "📍 Locais",
        "feat_roteiros": "🗺️ Roteiros",
        "info_objective": "Capacidades do Sistema",
        "info_objective_text": "LISBOA (Lisbon Itinerary System Based On AI) é uma plataforma de ponta destinada a residentes e turistas. Integra perfeitamente APIs de mobilidade em tempo real, dados meteorológicos e um repositório dinâmico de eventos culturais para garantir recomendações personalizadas sempre atualizadas.",
        "info_data_sources": "Redes Integradas",
        "info_data_sources_text": """- **API IPMA** - Atualizações meteorológicas
- **Metro de Lisboa** - Tempos de espera e estado
- **Carris & Carris Metropolitana** - Posições GPS e paragens
- **CP (Comboios de Portugal)** - Horários e serviços
- **Lisboa Aberta** - Dados essenciais da cidade
- **VisitLisboa** - Hub oficial de turismo""",
        "info_privacy": "Privacidade e Segurança",
        "info_privacy_text": "- As suas credenciais API são guardadas localmente apenas na sua sessão\n- Nenhum dado de conversa é guardado permanentemente\n- Operações de geolocalização descartadas após o uso",
        "info_how_to_use": "Como Utilizar",
        "info_how_to_use_text": """1. **Selecione o Motor de IA** - Escolha entre OpenAI, Azure ou LM Studio
 2. **Introduza as credenciais** - Forneça a chave API ou URL do servidor
 3. **Faça perguntas** - Escreva as suas perguntas em linguagem natural
 4. **Use Ações Rápidas** - Clique nos botões da barra lateral para consultas frequentes""",
        "info_author": "Autor",
        "info_author_text": """**André Filipe Gomes Silvestre**
Mestrando em Data Science e Advanced Analytics
NOVA IMS - Universidade NOVA de Lisboa
2025/2026""",
    },
}


def t(key: str) -> str:
    lang = st.session_state.get("language", "pt")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


def md_to_html(text: str) -> str:
    """Convert markdown bold syntax to HTML <strong> for use in unsafe_allow_html blocks."""
    return re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)


def rich_text_to_html(text: str) -> str:
    """Convert simple translation markdown blocks into safe HTML for `st.html()` sections."""
    if not text:
        return ""

    blocks: list[str] = []
    list_items: list[str] = []
    list_tag: Optional[str] = None

    def flush_list() -> None:
        nonlocal list_items, list_tag
        if not list_items or not list_tag:
            return
        items_html = "".join(f"<li>{item}</li>" for item in list_items)
        blocks.append(f"<{list_tag}>{items_html}</{list_tag}>")
        list_items = []
        list_tag = None

    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            flush_list()
            continue

        ordered_match = re.match(r"^\d+\.\s+(.*)$", line)
        unordered_match = re.match(r"^-\s+(.*)$", line)

        if ordered_match:
            content = md_to_html(html.escape(ordered_match.group(1).strip()))
            if list_tag != "ol":
                flush_list()
                list_tag = "ol"
            list_items.append(content)
            continue

        if unordered_match:
            content = md_to_html(html.escape(unordered_match.group(1).strip()))
            if list_tag != "ul":
                flush_list()
                list_tag = "ul"
            list_items.append(content)
            continue

        flush_list()
        blocks.append(f"<p>{md_to_html(html.escape(line))}</p>")

    flush_list()
    return "".join(blocks)


def render_html_block(content: str) -> None:
    """Render raw HTML reliably, preferring `st.html()` when available."""
    if hasattr(st, "html"):
        st.html(content)
        return
    st.markdown(content, unsafe_allow_html=True)


def build_info_feature_card_html(icon: str, title: str, description: str) -> str:
    """Build a feature card used on the Info page."""
    return (
        '<div class="info-card">'
        f'<div class="info-card-icon">{html.escape(icon)}</div>'
        f'<div class="info-card-title">{html.escape(title)}</div>'
        f'<div class="info-card-desc">{html.escape(description)}</div>'
        '</div>'
    )


def build_info_detail_card_html(icon: str, title: str, body: str) -> str:
    """Build a detail card used on the Info page."""
    return (
        '<div class="info-detail-card">'
        f'<div class="info-detail-title">{html.escape(icon)} <span>{html.escape(title)}</span></div>'
        f'<div class="info-detail-body">{body}</div>'
        '</div>'
    )


# ==========================================================================
# PRODUCTION UI - CUSTOM CSS AND ASSETS
# ==========================================================================


@st.cache_data(show_spinner=False)
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""


# Auto load assets
banner_path = os.path.join(os.path.dirname(__file__), "img", "BannerLSIBOA_21-9.png")
logo_path = os.path.join(os.path.dirname(__file__), "img", "Logo_1-1_WithoutBG.png")

banner_b64 = get_base64_image(banner_path)
logo_b64 = get_base64_image(logo_path)

banner_url = f"data:image/png;base64,{banner_b64}" if banner_b64 else ""
logo_url = f"data:image/png;base64,{logo_b64}" if logo_b64 else ""

CSS = f"""
<style>
/* ==========================================================================
   PRODUCTION CSS - HIGH-END UI
   Colors: Lisbon Yellow #f6da00, Lisbon Red/Orange #ff4011
   ========================================================================== */
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {{
    --primary-yellow: #f6da00;
    --primary-red: #ff4011;
    --dark-bg: #1a1a1a;
    --light-bg: #ffffff;
    --gray-50: #f8fafc;
    --gray-100: #e9eef5;
    --border-color: rgba(43, 43, 43, 0.1);
    --text-main: #2b2b2b;
    --text-muted: #5e5e5e;
    --shadow-sm: 0 10px 28px rgba(15, 23, 42, 0.07);
    --shadow-md: 0 18px 44px rgba(255, 64, 17, 0.16);
}}

/* Base Fonts */
html, body, [class*="css"]  {{
    font-family: 'Inter', sans-serif;
}}
h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
    font-family: 'Montserrat', sans-serif !important;
}}

/* Container width & layout */
.main .block-container {{
    max-width: none;
    width: 100%;
    padding: 1.5rem 2.25rem 2rem 2.25rem;
}}

/* Hide standard Streamlit header & footer for production */
header[data-testid="stHeader"] {{ background: transparent; box-shadow: none; }}
footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}

/* Custom Banner Component */
.top-banner-container {{
    background: linear-gradient(135deg, rgba(255, 64, 17, 0.15) 0%, rgba(255, 107, 71, 0.35) 30%, rgba(246, 218, 0, 0.35) 100%), url('{banner_url}');
    background-size: cover;
    background-position: center;
    border-radius: 20px;
    padding: 4rem 3.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(255, 64, 17, 0.25), 0 2px 8px rgba(0,0,0,0.1);
    position: relative;
    overflow: hidden;
}}

/* Decorative circles like app.py */
.top-banner-container::before {{
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: rgba(255,255,255,0.1);
    border-radius: 50%;
    pointer-events: none;
}}

.top-banner-container::after {{
    content: '';
    position: absolute;
    bottom: -30%;
    left: 10%;
    width: 150px;
    height: 150px;
    background: rgba(255,255,255,0.08);
    border-radius: 50%;
    pointer-events: none;
}}

.top-banner-container h1 {{
    color: white;
    margin: 0;
    font-size: 4.8rem;
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    letter-spacing: -0.04em;
    position: relative;
    z-index: 1;
}}

.top-banner-container p {{
    color: rgba(255,255,255,0.95);
    font-size: 1.15rem;
    font-weight: 400;
    position: relative;
    z-index: 1;
}}

/* Sidebar specific aesthetics */
[data-testid="stSidebar"] {{
    background: #fafafa !important;
    border-right: 1px solid #eee;
}}

.sidebar-logo {{
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
    margin-top: -3rem;
    padding: 1rem;
}}
.sidebar-logo img {{
    max-width: 180px;
    drop-shadow: 0px 5px 15px rgba(0,0,0,0.1);
}}

/* Buttons */
button {{
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif;
    transition: all 0.3s ease !important;
}}

button[kind="primary"] {{
    background: linear-gradient(135deg, var(--primary-red) 0%, #ff6a45 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0px 4px 12px rgba(255, 64, 17, 0.3) !important;
}}

button[kind="primary"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0px 6px 16px rgba(255, 64, 17, 0.4) !important;
}}

button[kind="secondary"] {{
    background: white !important;
    border: 1px solid #ddd !important;
    color: var(--text-main) !important;
}}

button[kind="secondary"]:hover {{
    border-color: var(--primary-yellow) !important;
    background: #fffcf0 !important;
}}

/* Chat Inputs */
[data-testid="stChatInput"] input {{
    font-family: 'Inter', sans-serif !important;
}}
[data-testid="stChatInput"] > div {{
    border-radius: 16px !important;
    border: 1.5px solid #dedede !important;
    transition: all 0.3s ease;
    box-shadow: var(--shadow-sm);
}}
[data-testid="stChatInput"] > div:focus-within {{
    border-color: var(--primary-red) !important;
    box-shadow: 0 0 0 4px rgba(255, 64, 17, 0.1) !important;
}}

/* Chat text spacing */
[data-testid="stChatMessage"] p {{
    margin-bottom: 0.3rem;
    line-height: 1.5;
}}

/* Bold text in chat - accent color */
[data-testid="stChatMessage"] strong {{
    color: var(--text-main);
    font-weight: 700;
}}

/* Headers in chat */
[data-testid="stChatMessage"] h3 {{
    color: var(--text-main);
    font-weight: 700;
    margin-top: 0.1rem;
    margin-bottom: 0.45rem;
    font-size: 1.2rem;
    line-height: 1.2;
}}
[data-testid="stChatMessage"] h4 {{
    margin-top: 0.5rem;
    margin-bottom: 0.2rem;
    font-size: 0.95rem;
}}

/* Hide Streamlit heading anchors inside chat response titles */
[data-testid="stChatMessage"] h1 a,
[data-testid="stChatMessage"] h2 a,
[data-testid="stChatMessage"] h3 a,
[data-testid="stChatMessage"] h4 a,
[data-testid="stChatMessage"] h5 a,
[data-testid="stChatMessage"] h6 a {{
    display: none !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}}

/* Horizontal rules - compact */
[data-testid="stChatMessage"] hr {{
    margin: 0.5rem 0;
}}

/* Links - pill-style with accent background */
[data-testid="stChatMessage"] p a,
[data-testid="stChatMessage"] li a,
[data-testid="stChatMessage"] td a {{
    color: var(--primary-red);
    text-decoration: none;
    font-weight: 600;
    padding: 0.15rem 0.4rem;
    border-radius: 6px;
    background: rgba(255, 64, 17, 0.08);
    border-bottom: 1.5px solid rgba(255, 64, 17, 0.3);
    transition: all 0.25s ease;
    display: inline;
}}
[data-testid="stChatMessage"] p a:hover,
[data-testid="stChatMessage"] li a:hover,
[data-testid="stChatMessage"] td a:hover {{
    color: var(--primary-red);
    background: rgba(255, 64, 17, 0.16);
    border-bottom-color: var(--primary-red);
    box-shadow: 0 2px 6px rgba(255, 64, 17, 0.2);
}}

/* Google Maps links keep the same pill shape but switch to the requested blue-green gradient */
[data-testid="stChatMessage"] a[href*="google.com/maps"] {{
    color: #14532d;
    background: linear-gradient(135deg, rgba(14, 224, 113, 0.2) 0%, rgba(55, 119, 255, 0.2) 100%);
    border-bottom-color: rgba(55, 119, 255, 0.45);
    box-shadow: 0 2px 8px rgba(55, 119, 255, 0.16);
}}
[data-testid="stChatMessage"] a[href*="google.com/maps"]:hover {{
    color: #0f172a;
    background: linear-gradient(135deg, rgba(14, 224, 113, 0.3) 0%, rgba(55, 119, 255, 0.28) 100%);
    border-bottom-color: rgba(14, 224, 113, 0.7);
    box-shadow: 0 4px 10px rgba(14, 224, 113, 0.2), 0 4px 12px rgba(55, 119, 255, 0.18);
}}

/* Unordered lists - clean with emoji support */
[data-testid="stChatMessage"] ul {{
    list-style-type: none;
    padding-left: 1.5rem;
    line-height: 1.45;
    margin-top: 0.2rem;
    margin-bottom: 0.3rem;
}}
[data-testid="stChatMessage"] ul ul {{
    padding-left: 1.2rem;
    margin-top: 0.1rem;
    margin-bottom: 0.15rem;
}}
[data-testid="stChatMessage"] ul li {{
    margin-bottom: 0.35rem;
    position: relative;
}}
[data-testid="stChatMessage"] ul ul li {{
    margin-bottom: 0.22rem;
}}
[data-testid="stChatMessage"] ul li::before {{
    content: "";
}}

/* Features Grid */
.features-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 15px;
    margin: 30px 0;
}}
.feature-card {{
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: var(--shadow-sm);
    border-top: 4px solid var(--primary-yellow);
    transition: transform 0.3s ease;
}}
.feature-card:nth-child(even) {{
    border-top: 4px solid var(--primary-red);
}}
.feature-card:hover {{
    transform: translateY(-5px);
    box-shadow: var(--shadow-md);
}}
.feature-card div {{
    font-weight: 600;
    font-size: 1rem;
    color: var(--text-main);
    margin-bottom: 8px;
}}
.feature-card p {{
    margin: 0;
    font-size: 0.9rem;
    color: var(--text-muted);
}}

/* Minimalist Expanders */
.streamlit-expanderHeader {{
    font-weight: 600;
    background: transparent !important;
    border: none !important;
    color: var(--text-muted) !important;
}}

/* Toast Alerts */
[data-testid="stToast"] {{
    background: #fff;
    border-left: 4px solid var(--primary-yellow);
    border-radius: 10px;
}}

/* ============ ALERT BOXES ============ */
.stSuccess {{
    background: linear-gradient(135deg, rgba(14, 224, 113, 0.1) 0%, rgba(14, 224, 113, 0.05) 100%) !important;
    border: 1px solid #0ee071 !important;
    border-radius: 12px !important;
}}
.stWarning {{
    background: linear-gradient(135deg, rgba(246, 218, 0, 0.1) 0%, rgba(246, 218, 0, 0.05) 100%) !important;
    border: 1px solid var(--primary-yellow) !important;
    border-radius: 12px !important;
}}
.stError {{
    background: linear-gradient(135deg, rgba(255, 64, 17, 0.1) 0%, rgba(255, 64, 17, 0.05) 100%) !important;
    border: 1px solid var(--primary-red) !important;
    border-radius: 12px !important;
}}

/* ============ SIDEBAR FOOTER ============ */
.sidebar-footer {{
    text-align: center;
    padding: 1.5rem 1rem;
    margin-top: 3rem;
    background: rgba(0, 0, 0, 0.03);
    border-radius: 12px;
    border: 1px solid rgba(0, 0, 0, 0.08);
}}
.sidebar-footer-version {{
    color: var(--text-main);
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.4rem;
}}
.sidebar-footer-made {{
    color: var(--text-muted);
    font-size: 0.8rem;
    line-height: 1.4;
}}

/* ============ INFO PAGE SECTIONS ============ */
.info-section {{
    background: white;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin: 1.25rem 0;
    border: none;
    border-left: 4px solid var(--primary-red);
    box-shadow: var(--shadow-sm);
}}
.info-section h3 {{
    color: var(--text-main);
    margin: 0 0 0.25rem 0;
    font-size: 1.2rem;
    font-weight: 600;
}}

/* ============ SCROLLBAR ============ */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}
::-webkit-scrollbar-track {{
    background: #f4f4f5;
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb {{
    background: #d4d4d8;
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: #a1a1aa;
}}

/* ============ ORDERED LISTS IN CHAT ============ */
[data-testid="stChatMessage"] ol {{
    padding-left: 0;
    list-style: none;
    counter-reset: item;
}}
[data-testid="stChatMessage"] ol > li {{
    counter-increment: item;
    margin-bottom: 1rem;
    position: relative;
    padding-left: 2.2rem;
}}
[data-testid="stChatMessage"] ol > li::before {{
    content: counter(item) ".";
    position: absolute;
    left: 0;
    font-weight: 700;
    color: var(--primary-red);
    font-size: 1.05em;
}}

</style>
"""

# ==========================================================================
# SYSTEM CORE
# ==========================================================================

st.set_page_config(
    page_title="LISBOA | Intelligent Tour & Urban System",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_meta_description() -> None:
    """Ensure a stable meta description exists in the document head for SEO audits.

    Uses ``streamlit.components.v1.html`` so the injected ``<script>`` actually
    executes inside an iframe sibling that mutates the parent document head.
    ``st.html`` sanitises script tags in current Streamlit releases, so it
    cannot perform DOM mutations.
    """

    description = (
        "LISBOA is a bilingual AI assistant for Lisbon weather, transport, events, "
        "places, and itinerary planning across the Lisbon Metropolitan Area."
    )
    # Imported lazily so test collection (which runs ``import app``) does not
    # require a live Streamlit runtime to evaluate this side-effect helper.
    from streamlit.components.v1 import html as components_html

    components_html(
        f"""
        <script>
        (() => {{
            const descriptionText = {json.dumps(description)};
            const doc = window.parent ? window.parent.document : window.document;
            let meta = doc.querySelector('meta[name="description"]');
            if (!meta) {{
                meta = doc.createElement('meta');
                meta.name = 'description';
                doc.head.appendChild(meta);
            }}
            if (meta.getAttribute('content') !== descriptionText) {{
                meta.setAttribute('content', descriptionText);
            }}
        }})();
        </script>
        """,
        height=0,
    )


inject_meta_description()


def normalized_value(value: Optional[str]) -> str:
    """Normalize optional text values loaded from env or UI."""
    if not isinstance(value, str):
        return str(value).strip() if value is not None else ""
    return value.strip()


def runtime_provider_selector_enabled() -> bool:
    """Return whether the sidebar should allow live provider switching."""
    return bool(getattr(Config, "ENABLE_PROVIDER_SELECTOR", True))


def runtime_credential_inputs_enabled() -> bool:
    """Return whether the sidebar should allow live credential editing."""
    return bool(getattr(Config, "ENABLE_PROVIDER_CREDENTIAL_INPUTS", True))


def runtime_settings_panel_visible() -> bool:
    """Return whether the settings panel should be visible in the sidebar."""
    return runtime_provider_selector_enabled() or runtime_credential_inputs_enabled()


def runtime_auto_initialize_enabled() -> bool:
    """Return whether the app should auto-initialize the assistant on startup."""
    return not runtime_settings_panel_visible()


def provider_configuration_hint(language: str) -> str:
    """Explain where credentials should be configured for the active runtime."""
    if runtime_credential_inputs_enabled():
        return (
            "Configure-o nas definições laterais."
            if language == "pt"
            else "Configure it in the sidebar."
        )
    return (
        "Configure-o nas variáveis de ambiente ou nos Streamlit secrets."
        if language == "pt"
        else "Configure it in environment variables or Streamlit secrets."
    )


def init_system_state():
    """Initialise session state with secure defaults."""
    defaults = {
        "messages": [],
        "assistant": None,
        "provider": Config.MODEL_PROVIDER,
        "last_provider": Config.MODEL_PROVIDER,
        "language": "pt",
        "initialized": False,
        "error": None,
        "current_page": "chat",
        "credentials": {
            "openai": {"api_key": normalized_value(os.getenv("OPENAI_API_KEY", ""))},
            "azure": {
                "api_key": normalized_value(os.getenv("AZURE_OPENAI_API_KEY", "")),
                "endpoint": normalized_value(os.getenv("AZURE_OPENAI_ENDPOINT", "")),
                "model": normalized_value(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""))
                or normalized_value(Config.AZURE_OPENAI_DEPLOYMENT_NAME)
                    or normalized_value(Config.DEFAULT_GPT_MODEL_NAME),
            },
            "lmstudio": {
                "base_url": normalized_value(Config.LMSTUDIO_BASE_URL),
                "model": normalized_value(Config.LMSTUDIO_MODEL_NAME),
            },
        },
        "ui_api_key_values": {
            "openai": "",
            "azure_api_key": "",
            "azure_endpoint": "",
            "azure_model": "",
        },
        "startup_resources_attempted": False,
        "startup_resources_ok": None,
        "startup_resources_status": {},
        "transport_db_status": None,
        "startup_auto_init_attempted_provider": None,
        "startup_auto_init_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_resource(show_spinner=False)
def pre_warm_vector_store() -> bool:
    """Load the vector store once per server process."""
    return _pre_warm_vector_store_impl()


@st.cache_resource(show_spinner=False)
def prepare_transport_database() -> Tuple[bool, str]:
    """Prepare Carris GTFS database once per server process."""
    return _prepare_transport_database_impl()


@st.cache_resource(show_spinner=False)
def pre_warm_transport_networks() -> Dict[str, Any]:
    """Warm the static transport datasets required by first-turn routing.

    The goal is to avoid the first user prompt paying the cold-start cost for
    Metro station cache loading, CP GTFS DB creation/checks, and Carris
    Metropolitana stop/line/route cache downloads.
    """
    return _pre_warm_transport_networks_impl()


def _run_startup_preload(language: str = "pt") -> Dict[str, Any]:
    """Load one-time shared resources needed by the production app."""
    return _run_startup_preload_impl(
        language=language,
        use_multi_agent=Config.USE_MULTI_AGENT,
    )


def ensure_startup_resources(
    show_spinner: bool = True,
    force_retry: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Ensure one-time shared resources are loaded during app startup."""
    attempted = bool(st.session_state.get("startup_resources_attempted", False))
    cached_ok = st.session_state.get("startup_resources_ok")
    cached_status = st.session_state.get("startup_resources_status") or {}

    if attempted and cached_ok is not None and not force_retry:
        return bool(cached_ok), cached_status

    language = st.session_state.get("language", "pt")
    spinner_text = (
        "🚀 A preparar conhecimento e dados de mobilidade..."
        if language == "pt"
        else "🚀 Preparing knowledge base and mobility data..."
    )

    def _load() -> Tuple[bool, Dict[str, Any]]:
        preload_status = _run_startup_preload(language)
        st.session_state.startup_resources_attempted = True
        st.session_state.startup_resources_ok = preload_status.get("ok")
        st.session_state.startup_resources_status = preload_status
        st.session_state.transport_db_status = preload_status.get("transport_status")
        return bool(preload_status.get("ok")), preload_status

    if show_spinner:
        with st.spinner(spinner_text):
            return _load()
    return _load()


def set_credentials_env(provider: str) -> None:
    """Apply stored credentials to environment variables and runtime config."""
    creds = st.session_state.credentials
    Config.MODEL_PROVIDER = provider

    openai_key = normalized_value(creds["openai"].get("api_key"))
    azure_key = normalized_value(creds["azure"].get("api_key"))
    azure_endpoint = normalized_value(creds["azure"].get("endpoint"))
    azure_model = normalized_value(creds["azure"].get("model")) or normalized_value(
        Config.AZURE_OPENAI_DEPLOYMENT_NAME
    ) or normalized_value(Config.DEFAULT_GPT_MODEL_NAME)
    lmstudio_url = normalized_value(creds["lmstudio"].get("base_url"))
    lmstudio_model = normalized_value(creds["lmstudio"].get("model"))

    if provider == "openai" and openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        Config.OPENAI_API_KEY = openai_key
    elif provider == "azure" and azure_key:
        os.environ["AZURE_OPENAI_API_KEY"] = azure_key
        os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = azure_model
        Config.AZURE_OPENAI_API_KEY = azure_key
        Config.AZURE_OPENAI_ENDPOINT = azure_endpoint
        Config.AZURE_OPENAI_DEPLOYMENT_NAME = azure_model
    elif provider == "lmstudio":
        Config.LMSTUDIO_BASE_URL = lmstudio_url
        Config.LMSTUDIO_MODEL_NAME = lmstudio_model


def provider_has_required_credentials(provider: str) -> Tuple[bool, Optional[str]]:
    """Validate the minimum credential set needed for the selected provider."""
    lang = st.session_state.get("language", "pt")
    creds = st.session_state.credentials
    openai_key = normalized_value(creds["openai"].get("api_key"))
    azure_key = normalized_value(creds["azure"].get("api_key"))
    azure_endpoint = normalized_value(creds["azure"].get("endpoint"))
    azure_model = normalized_value(creds["azure"].get("model")) or normalized_value(
        Config.AZURE_OPENAI_DEPLOYMENT_NAME
    ) or normalized_value(Config.DEFAULT_GPT_MODEL_NAME)
    lmstudio_url = normalized_value(st.session_state.credentials["lmstudio"].get("base_url"))
    lmstudio_model = normalized_value(st.session_state.credentials["lmstudio"].get("model"))

    if provider == "openai" and not openai_key:
        return (
            False,
            f"Falta a chave da API OpenAI. {provider_configuration_hint(lang)}"
            if lang == "pt"
            else f"Missing OpenAI API key. {provider_configuration_hint(lang)}",
        )

    if provider == "azure":
        missing = []
        if not azure_key:
            missing.append("API Key")
        if not azure_endpoint:
            missing.append("Endpoint")
        if not azure_model:
            missing.append("Deployment Name")
        if missing:
            missing_str = ", ".join(missing)
            return (
                False,
                f"Faltam credenciais Azure OpenAI: {missing_str}."
                if lang == "pt"
                else f"Missing Azure OpenAI credentials: {missing_str}.",
            )

    if provider == "lmstudio":
        if not lmstudio_url:
            return (
                False,
                "Falta o URL do servidor LM Studio."
                if lang == "pt"
                else "Missing LM Studio server URL.",
            )
        if not lmstudio_model:
            return (
                False,
                "Falta o nome do modelo LM Studio."
                if lang == "pt"
                else "Missing LM Studio model name.",
            )

    return True, None


def sanitize_backend_error(raw_error: str) -> str:
    """Redact obvious secrets and endpoints from backend error messages."""
    sanitized = re.sub(r"https?://[^\s'\"]+", "[URL_REDACTED]", raw_error)
    sanitized = re.sub(r"(sk-[A-Za-z0-9]{6})[A-Za-z0-9_-]+", r"\1...[REDACTED]", sanitized)
    sanitized = re.sub(r"(Bearer\s+)[^\s'\"]+", r"\1[REDACTED]", sanitized)
    return sanitized


def test_assistant_connection(provider: str) -> Tuple[bool, Optional[str]]:
    """Run a minimal inference request to confirm the selected model is ready."""
    lang = st.session_state.get("language", "pt")
    placeholder = st.empty()
    from agent.utils.model_connection_probe import perform_raw_model_connection_probe

    if Config.USE_MULTI_AGENT:
        test_llm = st.session_state.assistant.supervisor.llm
        model_display = st.session_state.assistant.model_name
    else:
        test_llm = getattr(st.session_state.assistant, "llm", None)
        model_display = getattr(st.session_state.assistant, "model_name", "Model")
        if test_llm is None:
            return True, None

    placeholder.info(
        f"🔄 A testar o modelo {model_display}..."
        if lang == "pt"
        else f"🔄 Testing model {model_display}..."
    )

    try:
        perform_raw_model_connection_probe(
            test_llm=test_llm,
            provider=provider,
            model_display=model_display,
        )

        placeholder.success(
            f"✅ Modelo pronto! ({model_display})"
            if lang == "pt"
            else f"✅ Model ready! ({model_display})"
        )
        placeholder.empty()
        return True, None
    except Exception as exc:
        placeholder.empty()
        sanitized_error = sanitize_backend_error(str(exc))

        if provider == "lmstudio":
            fixes = (
                "- Confirme que o LM Studio está aberto e com o servidor ativo\n"
                "- Verifique se o modelo selecionado está carregado\n"
                "- Confirme o URL e a porta do servidor local"
                if lang == "pt"
                else "- Make sure LM Studio is open and its local server is running\n"
                "- Confirm the selected model is fully loaded\n"
                "- Verify the server URL and port"
            )
        elif provider == "azure":
            fixes = (
                "- Verifique a API key, o endpoint e o deployment\n"
                "- Confirme que o deployment existe e está disponível\n"
                "- Valide quotas e permissões da subscrição"
                if lang == "pt"
                else "- Verify the API key, endpoint, and deployment\n"
                "- Confirm the deployment exists and is available\n"
                "- Validate subscription quotas and permissions"
            )
        else:
            fixes = (
                "- Verifique a API key\n- Confirme que o modelo está disponível\n- Valide a conectividade à Internet"
                if lang == "pt"
                else "- Verify the API key\n- Confirm the model is available\n- Check internet connectivity"
            )

        message = (
            "Não foi possível validar a ligação ao modelo.\n\n"
            f"{fixes}\n\nDetalhe técnico: {sanitized_error}"
            if lang == "pt"
            else "Could not validate the connection to the selected model.\n\n"
            f"{fixes}\n\nTechnical detail: {sanitized_error}"
        )
        return False, message


def initialize_assistant(
    provider: str,
    run_connection_probe: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Initialise the assistant securely and only when needed."""
    lang = st.session_state.get("language", "pt")
    credentials_ok, credentials_error = provider_has_required_credentials(provider)
    if not credentials_ok:
        st.session_state.initialized = False
        return False, credentials_error

    try:
        from agent.graph import MultiAgentAssistant, create_assistant

        set_credentials_env(provider)

        startup_ok, startup_status = ensure_startup_resources(
            show_spinner=False,
            force_retry=bool(st.session_state.get("startup_resources_attempted"))
            and not bool(st.session_state.get("startup_resources_ok")),
        )
        transport_ok = bool(startup_status.get("transport_ok", False))
        transport_status = str(
            startup_status.get("transport_status")
            or st.session_state.get("transport_db_status")
            or ""
        )
        st.session_state.transport_db_status = transport_status

        if Config.USE_MULTI_AGENT and not bool(startup_status.get("kb_ok", False)):
            st.session_state.initialized = False
            return (
                False,
                startup_status.get("kb_status")
                or (
                    "Não foi possível carregar a base de conhecimento."
                    if lang == "pt"
                    else "Could not load the knowledge base."
                ),
            )

        with st.spinner(
            "🤖 A iniciar o assistente..."
            if lang == "pt"
            else "🤖 Initializing assistant..."
        ):
            if Config.USE_MULTI_AGENT:
                st.session_state.assistant = MultiAgentAssistant()
            else:
                st.session_state.assistant = create_assistant(provider)

        if run_connection_probe:
            connection_ok, connection_error = test_assistant_connection(provider)
            if not connection_ok:
                st.session_state.assistant = None
                st.session_state.initialized = False
                return False, connection_error

        st.session_state.initialized = True
        st.session_state.provider = provider
        st.session_state.error = None

        if not transport_ok:
            st.toast(transport_status, icon="⚠️")

        return True, None
    except Exception as exc:
        st.session_state.assistant = None
        st.session_state.initialized = False
        st.session_state.error = sanitize_backend_error(str(exc))
        return (
            False,
            "Não foi possível iniciar o assistente."
            if lang == "pt"
            else "Could not initialise the assistant.",
        )


# ==========================================================================
# UI COMPONENTS
# ==========================================================================

if logo_path:
    st.logo("img/t.png", icon_image=logo_path, size="small")


def display_banner():
    st.markdown(
        f"""
        <div class="top-banner-container">
            <h1>{t("app_title")}</h1>
            <p>{t("app_subtitle")}</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

# def render_tracing_panel() -> None:
#     """Render LangSmith tracing status for the production sidebar."""
#     st.markdown(f"#### 🧭 {t('tracing')}")
#
#     tracing_display = get_langsmith_display_state()
#     langsmith_project = get_langsmith_project_name()
#
#     if tracing_display["state"] == "active":
#         st.success(t("tracing_active"))
#         st.caption(f"{t('project')}: {langsmith_project}")
#         return
#
#     if tracing_display["state"] == "auto_disabled_invalid_credentials":
#         st.warning(t("tracing_auto_disabled_invalid_credentials"))
#     elif tracing_display["state"] == "auto_disabled_invalid_configuration":
#         st.warning(t("tracing_auto_disabled_invalid_configuration"))
#     elif tracing_display["state"].startswith("auto_disabled"):
#         st.warning(t("tracing_auto_disabled"))
#     else:
#         st.warning(t("tracing_disabled"))
#
#     if tracing_display["state"].startswith("auto_disabled") and tracing_display.get("reason"):
#         st.caption(f"{t('tracing_reason')}: {tracing_display['reason']}")
#


def build_sidebar():
    with st.sidebar:
        request_locked = request_capture_locked(st.session_state.get("pending_request"))

        # Show custom Logo if exists
        if logo_url:
            st.markdown(
                f'<div class="sidebar-logo"><img src="{logo_url}" alt="LISBOA Logo" /></div>',
                unsafe_allow_html=True,
            )

        col1, col2 = st.columns(2)
        if col1.button(
            "🗺️ Chat",
            use_container_width=True,
            type="primary" if st.session_state.current_page == "chat" else "secondary",
        ):
            st.session_state.current_page = "chat"
            st.rerun()
        if col2.button(
            "ℹ️ Info",
            use_container_width=True,
            type="primary" if st.session_state.current_page == "info" else "secondary",
        ):
            st.session_state.current_page = "info"
            st.rerun()

        st.divider()

        # Simple Language Selection
        cur_lang = st.session_state.language
        langs = {"pt": "🇵🇹 Português", "en": "🇬🇧 English"}
        lang_idx = 0 if cur_lang == "pt" else 1
        new_lang_key = st.selectbox(
            t("language"),
            options=list(langs.keys()),
            format_func=lambda x: langs[x],
            index=lang_idx,
        )
        if new_lang_key != cur_lang:
            st.session_state.language = new_lang_key
            st.rerun()

        provider_labels = {
            "openai": "OpenAI",
            "azure": "Azure OpenAI",
            "lmstudio": "LM Studio",
        }
        manual_connect_visible = runtime_settings_panel_visible()
        locked_provider = Config.MODEL_PROVIDER
        if not runtime_provider_selector_enabled():
            st.session_state.provider = locked_provider
            st.session_state.last_provider = locked_provider
            selected_provider = locked_provider
        else:
            selected_provider = st.session_state.provider

        if manual_connect_visible:
            st.divider()

            with st.expander(
                "⚙️ " + t("settings"), expanded=False
            ):
                if runtime_provider_selector_enabled():
                    selected_provider = st.selectbox(
                        t("select_provider"),
                        options=list(provider_labels.keys()),
                        format_func=lambda key: provider_labels[key],
                        index=list(provider_labels.keys()).index(st.session_state.provider),
                    )

                credentials_changed = False
                if runtime_credential_inputs_enabled() and selected_provider == "openai":
                    if st.session_state.credentials["openai"].get("api_key"):
                        st.caption(
                            "🔐 Chave OpenAI detetada no ambiente. O valor nunca é mostrado."
                            if st.session_state.language == "pt"
                            else "🔐 OpenAI key detected in the environment. The value is never shown."
                        )

                    ui_value = st.session_state.ui_api_key_values.get("openai", "")
                    new_value = st.text_input(
                        "OpenAI API Key",
                        value=ui_value,
                        type="password",
                        placeholder=t("api_key_placeholder"),
                    )
                    if new_value != ui_value:
                        st.session_state.ui_api_key_values["openai"] = new_value
                        st.session_state.credentials["openai"]["api_key"] = new_value
                        credentials_changed = True

                elif runtime_credential_inputs_enabled() and selected_provider == "azure":
                    configured_items = []
                    if st.session_state.credentials["azure"].get("api_key"):
                        configured_items.append("API Key")
                    if st.session_state.credentials["azure"].get("endpoint"):
                        configured_items.append("Endpoint")
                    effective_azure_model = (
                        normalized_value(st.session_state.credentials["azure"].get("model"))
                        or normalized_value(Config.AZURE_OPENAI_DEPLOYMENT_NAME)
                        or normalized_value(Config.DEFAULT_GPT_MODEL_NAME)
                    )
                    if effective_azure_model:
                        configured_items.append("Deployment")
                    if configured_items:
                        configured_text = ", ".join(configured_items)
                        st.caption(
                            f"🔐 Configurado no ambiente: {configured_text}. Os valores nunca são mostrados."
                            if st.session_state.language == "pt"
                            else f"🔐 Configured in the environment: {configured_text}. Values are never shown."
                        )

                    ui_key = st.session_state.ui_api_key_values.get("azure_api_key", "")
                    ui_endpoint = st.session_state.ui_api_key_values.get("azure_endpoint", "")
                    ui_model = st.session_state.ui_api_key_values.get("azure_model", "")

                    new_key = st.text_input(
                        "Azure API Key",
                        value=ui_key,
                        type="password",
                        placeholder="Insira a chave Azure OpenAI..."
                        if st.session_state.language == "pt"
                        else "Enter your Azure OpenAI key...",
                    )
                    new_endpoint = st.text_input(
                        "Azure Endpoint",
                        value=ui_endpoint,
                        placeholder="https://your-resource.openai.azure.com",
                    )
                    new_model = st.text_input(
                        "Deployment Name",
                        value=ui_model,
                        placeholder=effective_azure_model,
                    )

                    if new_key != ui_key:
                        st.session_state.ui_api_key_values["azure_api_key"] = new_key
                        st.session_state.credentials["azure"]["api_key"] = new_key
                        credentials_changed = True
                    if new_endpoint != ui_endpoint:
                        st.session_state.ui_api_key_values["azure_endpoint"] = new_endpoint
                        st.session_state.credentials["azure"]["endpoint"] = new_endpoint
                        credentials_changed = True
                    if new_model != ui_model:
                        st.session_state.ui_api_key_values["azure_model"] = new_model
                        st.session_state.credentials["azure"]["model"] = new_model
                        credentials_changed = True

                elif runtime_credential_inputs_enabled():
                    current_base_url = st.session_state.credentials["lmstudio"].get(
                        "base_url", Config.LMSTUDIO_BASE_URL
                    )
                    current_model = st.session_state.credentials["lmstudio"].get(
                        "model", Config.LMSTUDIO_MODEL_NAME
                    )
                    new_base_url = st.text_input(
                        t("local_url"),
                        value=current_base_url,
                        placeholder=t("local_url_placeholder"),
                    )
                    new_model = st.text_input(
                        t("model_name"),
                        value=current_model,
                        placeholder=Config.LMSTUDIO_MODEL_NAME,
                    )
                    if new_base_url != current_base_url:
                        st.session_state.credentials["lmstudio"]["base_url"] = new_base_url
                        credentials_changed = True
                    if new_model != current_model:
                        st.session_state.credentials["lmstudio"]["model"] = new_model
                        credentials_changed = True

                if manual_connect_visible and st.button(
                    t("save_credentials"),
                    use_container_width=True,
                    type="primary",
                    key="connect_system_button",
                ):
                    with st.spinner(
                        "🔌 A ligar o assistente ao motor de IA..."
                        if st.session_state.language == "pt"
                        else "🔌 Connecting assistant to AI engine..."
                    ):
                        success, error = initialize_assistant(selected_provider)
                    if success:
                        st.session_state.startup_auto_init_attempted_provider = selected_provider
                        st.session_state.startup_auto_init_error = None
                        st.success(t("assistant_ready"))
                        st.rerun()
                    else:
                        st.session_state.startup_auto_init_attempted_provider = selected_provider
                        st.session_state.startup_auto_init_error = error or t("initialization_failed")
                        st.error(error or t("initialization_failed"))
                elif (
                    manual_connect_visible
                    and st.session_state.initialized
                    and st.session_state.provider == selected_provider
                    and not credentials_changed
                ):
                    st.success(t("assistant_ready"))
                else:
                    provider_ready, provider_msg = provider_has_required_credentials(
                        selected_provider
                    )
                    if provider_ready and manual_connect_visible:
                        st.info(
                            "Credenciais prontas. Clique em **Ligar Sistema** para iniciar."
                            if st.session_state.language == "pt"
                            else "Credentials are ready. Click **Connect System** to start."
                        )
                    elif provider_msg:
                        st.caption(provider_msg)

            st.divider()

        # Quick Actions
        st.markdown(f"#### ⚡ {t('quick_actions')}")
        quick_acts = [
            ("🌤️", t("weather_summary"), t("query_weather")),
            ("🚇", t("transport_status"), t("query_transport")),
            ("🎭", t("upcoming_events"), t("query_events")),
            ("📍", t("top_attractions"), t("query_attractions")),
            ("🗺️", t("plan_my_day"), t("query_plan")),
        ]

        q_act = None
        for idx, (icon, label, qt) in enumerate(quick_acts):
            if st.button(
                f"{icon} {label}",
                use_container_width=True,
                key=f"sidebar_qact_{idx}",
                disabled=request_locked,
            ):
                q_act = qt

        st.divider()

        # Session info
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric(t("messages"), count_user_interactions(st.session_state.messages))
        with col_s2:
            provider_ready, _ = provider_has_required_credentials(selected_provider)
            if st.session_state.initialized and st.session_state.provider == selected_provider:
                st.metric(t("status"), "🟢")
            elif provider_ready:
                st.metric(t("status"), "🟡")
            else:
                st.metric(t("status"), "⚪")

        if st.session_state.initialized and hasattr(st.session_state, "assistant") and st.session_state.assistant:
            model_name = getattr(st.session_state.assistant, "model_name", None)
            if model_name:
                st.caption(f"🤖 {model_name}")

        if st.session_state.messages:
            if st.button("🗑️ " + t("clear_conversation"), use_container_width=True):
                st.session_state.messages = []
                if st.session_state.assistant and hasattr(st.session_state.assistant, "reset"):
                    st.session_state.assistant.reset()
                st.rerun()

        # LangSmith tracing sidebar panel intentionally disabled in the UI.
        # render_tracing_panel()

        st.markdown(
            f"""
        <div class="sidebar-footer">
            <div class="sidebar-footer-version">{t("footer_version")}</div>
            <div class="sidebar-footer-made">{t("footer_made")}</div>
            <div class="sidebar-footer-made" style="margin-top:2px;">{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        return selected_provider, q_act


def build_welcome():
    st.markdown(
        f"<h2 style='text-align: center; margin-bottom: 10px;'>{t('welcome_title')}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align: center; color: var(--text-muted); font-size: 1.1rem;'>{t('welcome_intro')}</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="features-grid">
            <div class="feature-card"><div>{t("feat_atmosfera")}</div><p>{md_to_html(t('weather_desc'))}</p></div>
            <div class="feature-card"><div>{t("feat_mobilidade")}</div><p>{md_to_html(t('transport_desc'))}</p></div>
            <div class="feature-card"><div>{t("feat_cultura")}</div><p>{md_to_html(t('events_desc'))}</p></div>
            <div class="feature-card"><div>{t("feat_mapa")}</div><p>{md_to_html(t('places_desc'))}</p></div>
            <div class="feature-card"><div>{t("feat_roteiros")}</div><p>{md_to_html(t('planning_desc'))}</p></div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    examples = [
        ("🌤️", t("ex_weather"), t("ex_query_weather")),
        ("🚇", t("ex_metro"), t("ex_query_metro")),
        ("🎭", t("ex_events"), t("ex_query_events")),
        ("🏥", t("ex_services"), t("ex_query_services")),
        ("🍽️", t("ex_food"), t("ex_query_food")),
        ("🗺️", t("ex_planning"), t("ex_query_planning")),
    ]

    st.markdown(f"### 💡 {t('try_asking')}")
    cols = st.columns(3)
    chosen_ex = None
    for i, (ic, lab, qt) in enumerate(examples):
        with cols[i % 3]:
            if st.button(f"{ic} {lab}", use_container_width=True, key=f"exq_{i}"):
                chosen_ex = qt
    return chosen_ex


def handle_chat_stream(text: str):
    """Yield text in smart chunks for streaming display.

    Uses line-based chunking: emits complete lines at once so markdown
    renders correctly during streaming (no broken bold/links mid-line).
    Falls back to word chunks for very long paragraphs.
    """
    if not text:
        return
    lines = text.split("\n")
    for i, line in enumerate(lines):
        suffix = "\n" if i < len(lines) - 1 else ""
        # Short lines: emit whole line at once
        if len(line) <= 120:
            yield line + suffix
            time.sleep(0.03)
        else:
            # Long lines: emit in word groups of 6
            words = line.split(" ")
            buf = []
            for w in words:
                buf.append(w)
                if len(buf) >= 6:
                    yield " ".join(buf) + " "
                    buf = []
                    time.sleep(0.02)
            if buf:
                yield " ".join(buf) + suffix
                time.sleep(0.02)


def render_assistant_markdown(text: str) -> str:
    """Render assistant markdown progressively, then re-render the final full markdown."""
    if not text:
        st.markdown("")
        return ""

    placeholder = st.empty()
    rendered_chunks: list[str] = []

    for chunk in handle_chat_stream(text):
        rendered_chunks.append(chunk)
        placeholder.markdown("".join(rendered_chunks))

    # Force a clean final render from the original full text. During streaming,
    # partial markdown can momentarily create malformed list or heading HTML,
    # especially for long event/place cards. Replacing the placeholder ensures
    # Streamlit rebuilds the DOM from the canonical final markdown.
    final_text = text
    placeholder.empty()
    st.empty().markdown(final_text)
    return final_text


def clean_response_for_display(text: str) -> str:
    """Remove obvious citation artefacts before rendering the final response."""
    cleaned = re.sub(r"【.*?】", "", text or "")
    return cleaned.replace("\x00", "").strip()


def count_user_interactions(messages: list[dict[str, Any]]) -> int:
    """Count user turns only, so the sidebar metric reflects interactions, not message pairs."""
    return sum(
        1
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    )


def startup_gate_allows_requests(
    startup_ok: bool,
    startup_status: Dict[str, Any],
    *,
    use_multi_agent: bool,
) -> bool:
    """Return whether the startup readiness gate is open for new user requests."""
    if not startup_ok or not bool(startup_status.get("ok", False)):
        return False
    if not bool(startup_status.get("transport_ok", False)):
        return False
    if use_multi_agent and not bool(startup_status.get("kb_ok", False)):
        return False
    return True


def should_attempt_startup_auto_initialization(
    *,
    initialized: bool,
    current_provider: str,
    selected_provider: str,
    credentials_ready: bool,
    attempted_provider: Optional[str],
    last_error: Optional[str],
) -> bool:
    """Return whether the startup flow should auto-initialize the assistant."""
    if not runtime_auto_initialize_enabled() or not credentials_ready:
        return False
    if initialized and current_provider == selected_provider:
        return False
    if attempted_provider == selected_provider and last_error:
        return False
    return True


def build_startup_gate_message(
    startup_status: Dict[str, Any],
    *,
    language: str,
    use_multi_agent: bool,
) -> str:
    """Build a concise readiness message listing the startup checks that failed."""
    if language == "pt":
        lines = ["As verificações de arranque ainda não estão completas."]
    else:
        lines = ["Startup checks are incomplete."]

    transport_status = str(startup_status.get("transport_status") or "").strip()
    if not bool(startup_status.get("transport_ok", False)) and transport_status:
        lines.append(transport_status)

    kb_status = str(startup_status.get("kb_status") or "").strip()
    if use_multi_agent and not bool(startup_status.get("kb_ok", False)) and kb_status:
        lines.append(kb_status)

    return "\n\n".join(lines)


def select_new_request(
    *,
    sidebar_request: Optional[str],
    welcome_request: Optional[str],
    chat_request: Optional[str],
    pending_request: Optional[str],
    allow_requests: bool = True,
) -> Optional[str]:
    """Choose the next new request while preventing duplicate consumption across reruns."""
    if pending_request or not allow_requests:
        return None
    return chat_request or welcome_request or sidebar_request


def request_capture_locked(pending_request: Optional[str]) -> bool:
    """Return whether the UI should temporarily block new requests."""
    return bool(pending_request)


def build_user_error_message(error: Exception) -> str:
    """Convert backend exceptions into safe, user-friendly messages."""
    lang = st.session_state.get("language", "pt")
    error_str = str(error).lower()

    if "401" in error_str or "unauthorized" in error_str:
        return (
            "Falha de autenticação. Verifique a configuração da chave e do fornecedor."
            if lang == "pt"
            else "Authentication failed. Check the API key and provider settings."
        )
    if "rate" in error_str or "limit" in error_str:
        return (
            "O limite de pedidos foi atingido. Aguarde um momento e tente novamente."
            if lang == "pt"
            else "The request limit was reached. Please wait a moment and try again."
        )
    if "content_filter" in error_str or "responsibleaipolicyviolation" in error_str:
        return (
            "O fornecedor bloqueou temporariamente este pedido. Reformule a pergunta e tente novamente."
            if lang == "pt"
            else "The provider temporarily blocked this request. Please rephrase it and try again."
        )
    if "timeout" in error_str or "connection" in error_str:
        return (
            "Não foi possível contactar o fornecedor do modelo. Verifique a ligação ou o servidor local."
            if lang == "pt"
            else "Could not reach the selected model provider. Check your connection or local server."
        )
    return t("error_generic")


def run_interaction(user_input: str, user_message_already_rendered: bool = False):
    """Run one chat turn and optionally skip re-rendering the user message."""
    if not user_message_already_rendered:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            with st.status("🔍 " + t("thinking"), expanded=False) as status:
                last_status = {"label": "", "ts": 0.0}

                def on_status(msg: str):
                    normalized = str(msg or "").strip()
                    if not normalized:
                        return

                    now = time.perf_counter()
                    if normalized == last_status["label"]:
                        return
                    if last_status["label"] and (now - last_status["ts"]) < 0.12:
                        return

                    last_status["label"] = normalized
                    last_status["ts"] = now
                    status.update(label="⚡ " + normalized, state="running")

                resp = st.session_state.assistant.chat(
                    user_input,
                    verbose=False,
                    on_status_change=on_status,
                    language=st.session_state.language,
                )
                status.update(
                    label="✅ Resposta pronta!"
                    if st.session_state.language == "pt"
                    else "✅ Response ready!",
                    state="complete",
                )

            sanitized = clean_response_for_display(resp)
            rendered_response = render_assistant_markdown(sanitized)
            st.session_state.messages.append(
                {"role": "assistant", "content": rendered_response}
            )

        except Exception as error:
            # Log the FULL traceback to the terminal for debugging
            import traceback
            print(f"\n{'=' * 70}")
            print("ERROR during chat interaction:")
            print(f"  Query: {user_input}")
            print(f"  Error type: {type(error).__name__}")
            print(f"  Error message: {error}")
            print(f"{'=' * 70}")
            traceback.print_exc()
            print(f"{'=' * 70}\n")

            friendly_message = build_user_error_message(error)
            st.error(f"⚠️ {friendly_message}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ {friendly_message}"}
            )


def queue_pending_request(user_input: str) -> None:
    """Queue a new request and append the user turn before the next rerun.

    This keeps the sidebar interaction counter in sync on the first click or
    first submitted message instead of lagging one rerun behind.
    """
    if not user_input:
        return

    st.session_state.pending_request = user_input
    st.session_state.pending_request_user_appended = True
    st.session_state.messages.append({"role": "user", "content": user_input})


def run_info_page():
    render_html_block("""
        <style>
        .info-main-container { padding: 1rem 0; animation: fadeIn 0.8s ease; }
        .info-header { text-align: center; margin-bottom: 3.5rem; }
        .info-header h2 { font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, var(--primary-red) 0%, #ff6b6b 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; letter-spacing: -1px; }
        .info-header h4 { color: var(--text-muted); font-weight: 500; font-size: 1.15rem; margin-bottom: 1.5rem; }
        .info-header p { color: var(--text-main); font-size: 1.15rem; max-width: 820px; margin: 0 auto; line-height: 1.7; }

        .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.5rem; margin-bottom: 2.5rem; }
        .info-card { background: white; border: 1px solid var(--border-color); border-radius: 16px; padding: 2rem; transition: all 0.3s ease; box-shadow: var(--shadow-sm); position: relative; overflow: hidden; }
        .info-card::before { content: ""; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: var(--primary-red); opacity: 0.45; transition: opacity 0.3s ease; }
        .info-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-md); border-color: rgba(255, 64, 17, 0.28); }
        .info-card:hover::before { opacity: 1; }
        .info-card-icon { font-size: 2.2rem; margin-bottom: 1.2rem; display: inline-block; background: var(--gray-50); padding: 0.75rem 1rem; border-radius: 12px; }
        .info-card-title { font-weight: 700; font-size: 1.25rem; color: var(--text-main); margin-bottom: 0.75rem; }
        .info-card-desc { color: var(--text-muted); line-height: 1.65; font-size: 1rem; }

        .info-architecture { background: linear-gradient(120deg, #ffffff 0%, var(--gray-50) 100%); border: 1px solid var(--border-color); border-left: 5px solid var(--primary-red); border-radius: 18px; padding: 2.25rem; margin-bottom: 2rem; box-shadow: var(--shadow-sm); }
        .info-arch-title { font-weight: 800; font-size: 1.55rem; margin-bottom: 0.9rem; color: var(--text-main); display: flex; align-items: center; gap: 0.75rem; }
        .info-arch-desc { color: var(--text-main); font-size: 1.08rem; line-height: 1.75; }

        .info-details-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; margin-bottom: 2.25rem; }
        .info-detail-card { background: linear-gradient(180deg, #ffffff 0%, #fffdf8 100%); border: 1px solid var(--border-color); border-radius: 16px; padding: 1.6rem 1.7rem; box-shadow: var(--shadow-sm); }
        .info-detail-title { display: flex; align-items: center; gap: 0.6rem; font-weight: 800; color: var(--text-main); font-size: 1.08rem; margin-bottom: 0.9rem; }
        .info-detail-body { color: var(--text-main); line-height: 1.7; }
        .info-detail-body p { margin: 0 0 0.75rem 0; }
        .info-detail-body p:last-child { margin-bottom: 0; }
        .info-detail-body ul, .info-detail-body ol { margin: 0; padding-left: 1.25rem; }
        .info-detail-body li { margin-bottom: 0.45rem; color: var(--text-muted); }
        .info-detail-body strong { color: var(--text-main); }

        .info-footer { display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 1rem; padding-top: 2.2rem; border-top: 1px solid var(--border-color); }
        .info-author-box { background: var(--gray-50); border-radius: 16px; padding: 2rem 2.5rem; border: 1px solid var(--gray-100); text-align: center; max-width: 760px; box-shadow: var(--shadow-sm); }
        .info-author-label { text-transform: uppercase; font-size: 0.85rem; letter-spacing: 1.5px; color: var(--text-muted); margin-bottom: 0.8rem; display: block; font-weight: 700; }
        .info-author-text { color: var(--text-main); line-height: 1.75; font-size: 1.04rem; }
        .info-author-text p { margin: 0 0 0.5rem 0; }
        .info-author-text p:last-child { margin-bottom: 0; }

        .back-btn-container { margin-top: 3rem; display: flex; justify-content: center; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        </style>
    """)

    feature_cards = "".join(
        [
            build_info_feature_card_html("🏛️", t("info_f1_title"), t("info_f1_desc")),
            build_info_feature_card_html("🚇", t("info_f2_title"), t("info_f2_desc")),
            build_info_feature_card_html("🌤️", t("info_f3_title"), t("info_f3_desc")),
            build_info_feature_card_html("🏥", t("info_f4_title"), t("info_f4_desc")),
        ]
    )

    detail_cards = "".join(
        [
            build_info_detail_card_html("🎯", t("info_objective"), rich_text_to_html(t("info_objective_text"))),
            build_info_detail_card_html("🧩", t("info_data_sources"), rich_text_to_html(t("info_data_sources_text"))),
            build_info_detail_card_html("🔒", t("info_privacy"), rich_text_to_html(t("info_privacy_text"))),
            build_info_detail_card_html("🛠️", t("info_how_to_use"), rich_text_to_html(t("info_how_to_use_text"))),
        ]
    )

    html_content = (
        '<div class="info-main-container">'
        '<div class="info-header">'
        f'<h2>{html.escape(t("info_title"))}</h2>'
        f'<h4>{html.escape(t("info_subtitle"))}</h4>'
        f'<p>{html.escape(t("info_intro"))}</p>'
        '</div>'
        f'<div class="info-grid">{feature_cards}</div>'
        '<div class="info-architecture">'
        f'<div class="info-arch-title">⚙️ <span>{html.escape(t("info_architecture_title"))}</span></div>'
        f'<div class="info-arch-desc">{html.escape(t("info_architecture_desc"))}</div>'
        '</div>'
        f'<div class="info-details-grid">{detail_cards}</div>'
        '<div class="info-footer">'
        '<div class="info-author-box">'
        f'<span class="info-author-label">🎓 {html.escape(t("info_author"))}</span>'
        f'<div class="info-author-text">{rich_text_to_html(t("info_author_text"))}</div>'
        '</div>'
        '</div>'
        '</div>'
    )

    render_html_block(html_content)

    st.markdown('<div class="back-btn-container">', unsafe_allow_html=True)
    back_text = t("back_to_chat")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(f"💬 {back_text}", type="primary", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================================================
# MAIN EXECUTION
# ==========================================================================


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    init_system_state()

    ensure_startup_resources(
        show_spinner=not bool(st.session_state.get("startup_resources_attempted", False))
    )

    bootstrap_provider = (
        Config.MODEL_PROVIDER
        if not runtime_provider_selector_enabled()
        else st.session_state.get("provider", Config.MODEL_PROVIDER)
    )
    credentials_ready, _ = provider_has_required_credentials(bootstrap_provider)
    if should_attempt_startup_auto_initialization(
        initialized=bool(st.session_state.get("initialized", False)),
        current_provider=str(st.session_state.get("provider", "")),
        selected_provider=bootstrap_provider,
        credentials_ready=credentials_ready,
        attempted_provider=st.session_state.get("startup_auto_init_attempted_provider"),
        last_error=st.session_state.get("startup_auto_init_error"),
    ):
        success, error = initialize_assistant(bootstrap_provider)
        st.session_state.startup_auto_init_attempted_provider = bootstrap_provider
        st.session_state.startup_auto_init_error = (
            None if success else error or t("initialization_failed")
        )
        if success:
            st.session_state.error = None

    display_banner()
    selected_provider, q_act = build_sidebar()

    if st.session_state.current_page == "info":
        run_info_page()
        return

    pending = st.session_state.get("pending_request")
    request_locked = request_capture_locked(pending)

    # Stage 1: Capture a new request from quick-action, chat input, or welcome
    # button. Queue it and append the user turn immediately, then rerun once so
    # the sidebar counter and chat history both reflect the new turn before the
    # assistant call starts.
    welcome_request = None
    chat_request = None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not request_locked and not st.session_state.messages:
        welcome_request = build_welcome()

    if not request_locked and (in_text := st.chat_input(t("chat_placeholder"))):
        chat_request = in_text

    new_request = select_new_request(
        sidebar_request=q_act or None,
        welcome_request=welcome_request,
        chat_request=chat_request,
        pending_request=pending,
    )

    if new_request:
        queue_pending_request(new_request)
        st.rerun()
    
    # Stage 2: If a request is pending, ensure the assistant is initialized and
    # execute the LLM call. The user message has already been appended during
    # the previous rerun, so the sidebar counter is already in sync.
    pending = st.session_state.get("pending_request")
    if pending:
        if (
            not st.session_state.initialized
            or st.session_state.provider != selected_provider
        ):
            success, error = initialize_assistant(
                selected_provider,
                run_connection_probe=False,
            )
            if not success:
                st.error(error or t("initialization_failed"))
                return
        already_appended = bool(st.session_state.pop("pending_request_user_appended", False))
        st.session_state.pop("pending_request", None)
        run_interaction(pending, user_message_already_rendered=already_appended)
        # Trigger one final rerun so the sidebar counter picks up the assistant
        # turn immediately instead of waiting for the next user action.
        st.rerun()

    if not st.session_state.initialized:
        credentials_ready, _ = provider_has_required_credentials(selected_provider)
        auto_init_error = st.session_state.get("startup_auto_init_error")
        if runtime_auto_initialize_enabled() and auto_init_error:
            st.error(auto_init_error)
        elif credentials_ready and not runtime_auto_initialize_enabled():
            st.info(
                "As credenciais já estão prontas. Pode clicar em **Ligar Sistema** na barra lateral ou enviar uma pergunta para iniciar automaticamente."
                if st.session_state.language == "pt"
                else "Your credentials are ready. Click **Connect System** in the sidebar or send a prompt to start automatically."
            )
        else:
            st.info(
                "Configure as credenciais de produção nas variáveis de ambiente ou nos Streamlit secrets para começar."
                if st.session_state.language == "pt" and not runtime_credential_inputs_enabled()
                else "Configure the production credentials in environment variables or Streamlit secrets to get started."
                if not runtime_credential_inputs_enabled()
                else "Configure o fornecedor de IA nas definições laterais para começar."
                if st.session_state.language == "pt"
                else "Configure the AI provider in the sidebar settings to get started."
            )


if __name__ == "__main__":
    main()
