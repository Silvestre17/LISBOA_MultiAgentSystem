# ==========================================================================
# Master Thesis - Lisbon Urban Assistant (Streamlit App)
#   - Andre Filipe Gomes Silvestre, 20240502
# 
#   Main Streamlit application for the intelligent tourist assistant.
#   Provides a modern, intuitive chat interface for exploring Lisbon.
# 
#   Features:
#     - Real-time chat with LLM-powered assistant
#     - Multi-language UI support (English/Portuguese)
#     - Multiple LLM provider selection with credential management
#     - Weather and transport quick actions
#     - Session state management
#     - Professional Lisbon-themed design
# 
#   Usage:
#     streamlit run app.py
# ==========================================================================

# Required libraries:
# pip install streamlit langchain langgraph langchain-groq python-dotenv

# IMPORTANT: Load environment variables FIRST (before any LangChain imports)
from dotenv import load_dotenv
load_dotenv()

# Suppress Torch/Streamlit file watcher warning (known compatibility issue)
import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import streamlit as st
import sys
import os
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

# Add project root to path for imports
sys.path.insert(0, ".")

from agent.graph import create_assistant, LisbonAssistant, MultiAgentAssistant
from config import Config
from tools.visitlisboa_api import initialize_vector_store


# ==========================================================================
# TRANSLATIONS / INTERNATIONALIZATION
# ==========================================================================

TRANSLATIONS = {
    "en": {
        # Header
        "app_title": "Lisbon Urban Assistant",
        "app_subtitle": "Your intelligent guide to exploring Lisbon",
        
        # Sidebar - Settings
        "settings": "Settings",
        "language": "Language",
        "llm_provider": "LLM Provider",
        "select_provider": "Select AI Provider",
        "api_credentials": "API Credentials",
        "api_key": "API Key",
        "api_key_placeholder": "Enter your API key...",
        "local_url": "Local Server URL",
        "local_url_placeholder": "http://localhost:1234/v1",
        "model_name": "Model Name",
        "model_name_placeholder": "e.g., llama3.2",
        "save_credentials": "Save & Connect",
        "assistant_ready": "Assistant ready!",
        "initialization_failed": "Initialization failed",
        
        # Sidebar - Quick Actions
        "quick_actions": "Quick Actions",
        "weather_summary": "Weather Summary",
        "transport_status": "Transport Status",
        "upcoming_events": "Upcoming Events",
        "top_attractions": "Top Attractions",
        "plan_my_day": "Plan My Day",
        
        # Sidebar - Session Info
        "session_info": "Session Info",
        "messages": "Messages",
        "status": "Status",
        "clear_conversation": "Clear Conversation",
        
        # Sidebar - About
        "about": "About",
        "tracing": "Tracing",
        "tracing_active": "LangSmith Active",
        "tracing_disabled": "LangSmith Disabled",
        "project": "Project",
        
        # Main Content
        "welcome_title": "Welcome to Lisbon!",
        "welcome_intro": "I'm your intelligent assistant for exploring Lisbon, Portugal. I can help you with:",
        "weather_desc": "<strong>Weather</strong> - Current conditions and forecasts",
        "transport_desc": "<strong>Transport</strong> - Metro, bus, and train status",
        "events_desc": "<strong>Events</strong> - Cultural events and activities",
        "places_desc": "<strong>Places</strong> - Points of interest and services",
        "planning_desc": "<strong>Planning</strong> - Personalized itineraries",
        "ask_anything": "Ask me anything about Lisbon!",
        "try_asking": "Try asking about...",
        "chat_placeholder": "Ask me about Lisbon...",
        
        # Example Queries
        "ex_weather": "Weather",
        "ex_metro": "Metro",
        "ex_events": "Events",
        "ex_services": "Services",
        "ex_food": "Food",
        "ex_planning": "Planning",
        
        # Quick Action Queries
        "query_weather": "What's the current weather in Lisbon? Include any active warnings.",
        "query_transport": "What's the current status of public transport in Lisbon? Include Metro, buses, and trains.",
        "query_events": "What cultural events are happening in Lisbon this week?",
        "query_attractions": "What are the must-see tourist attractions in Lisbon?",
        "query_plan": "Help me plan a one-day trip in Lisbon. I'm interested in history and good food.",
        
        # Example Query Texts
        "ex_query_weather": "What's the weather forecast for the next 3 days in Lisbon?",
        "ex_query_metro": "Is the Lisbon metro running normally today?",
        "ex_query_events": "What cultural events are happening this weekend?",
        "ex_query_services": "Find pharmacies and hospitals near Rossio",
        "ex_query_food": "Recommend traditional Portuguese restaurants in Alfama",
        "ex_query_planning": "Plan a 2-day itinerary for a first-time visitor to Lisbon",
        
        # Errors
        "error_not_initialized": "Assistant Not Initialized",
        "error_troubleshooting": "Troubleshooting",
        "error_common_issues": "Common Issues:",
        "error_missing_api": "Missing API Key",
        "error_local_models": "Local Models (LM Studio / Ollama)",
        "error_network": "Network Issues",
        "retry_init": "Retry Initialization",
        "error_api_key": "API Key Error (401 Unauthorized)",
        "error_api_key_msg": "Your API key is invalid, expired, or revoked.",
        "error_rate_limit": "Rate Limit Exceeded",
        "error_rate_limit_msg": "You've exceeded the API rate limit. Please wait and try again.",
        "error_connection": "Connection Error",
        "error_connection_msg": "Could not connect to the API. Please check your internet connection.",
        "error_generic": "An error occurred while processing your request.",
        "thinking": "Analyzing and gathering information...",
        
        # Footer
        "footer_version": "Lisbon Urban Assistant v1.0",
        "footer_made": "André Filipe Gomes Silvestre | Master's Student\nNOVA IMS",
        
        # Info Page
        "info_title": "About This Assistant",
        "info_objective": "Objective",
        "info_objective_text": "This intelligent assistant was developed as part of a Master's Thesis in Data Science and Advanced Analytics at NOVA IMS (Universidade NOVA de Lisboa). The goal is to create an LLM-powered framework for adaptive tourist and mobility itinerary planning in Lisbon.",
        "info_data_sources": "Data Sources",
        "info_data_sources_text": """The assistant uses multiple real-time and static data sources:

- **IPMA API** - Weather forecasts and meteorological warnings
- **Metro de Lisboa** - Real-time status of all 4 metro lines
- **Carris Metropolitana** - Bus alerts, stops, and line information
- **CP (Comboios de Portugal)** - Train status and delays
- **Lisboa Aberta** - Open data (pharmacies, hospitals, museums, etc.)
- **VisitLisboa** - Cultural events, attractions, and points of interest
- **Official Lisbon Guide** - Tourist guide PDF with comprehensive city information""",
        "info_how_to_use": "How to Use",
        "info_how_to_use_text": """1. **Select your LLM Provider** - Choose from Groq, Google, OpenAI, or local models
2. **Enter your credentials** - Provide the required API key or server URL
3. **Ask questions** - Type your questions in natural language
4. **Use Quick Actions** - Click sidebar buttons for common queries""",
        "info_privacy": "Privacy & Security",
        "info_privacy_text": """- Your API credentials are stored locally in your browser session only
- No conversation data is stored permanently on any server
- LangSmith tracing (if enabled) is for development purposes only""",
        "info_author": "Author",
        "info_author_text": """**Andre Filipe Gomes Silvestre**
Master's Student in Data Science and Advanced Analytics
NOVA IMS - Universidade NOVA de Lisboa
2024/2025""",
    },
    
    "pt": {
        # Header
        "app_title": "Assistente Urbano de Lisboa",
        "app_subtitle": "O seu guia inteligente para explorar Lisboa",
        
        # Sidebar - Settings
        "settings": "Definições",
        "language": "Idioma",
        "llm_provider": "Fornecedor LLM",
        "select_provider": "Selecionar Fornecedor IA",
        "api_credentials": "Credenciais API",
        "api_key": "Chave API",
        "api_key_placeholder": "Introduza a sua chave API...",
        "local_url": "URL do Servidor Local",
        "local_url_placeholder": "http://localhost:1234/v1",
        "model_name": "Nome do Modelo",
        "model_name_placeholder": "ex: llama3.2",
        "save_credentials": "Guardar e Ligar",
        "assistant_ready": "Assistente pronto!",
        "initialization_failed": "Falha na inicialização",
        
        # Sidebar - Quick Actions
        "quick_actions": "Ações Rápidas",
        "weather_summary": "Resumo do Tempo",
        "transport_status": "Estado dos Transportes",
        "upcoming_events": "Próximos Eventos",
        "top_attractions": "Principais Atrações",
        "plan_my_day": "Planear o Meu Dia",
        
        # Sidebar - Session Info
        "session_info": "Info da Sessão",
        "messages": "Mensagens",
        "status": "Estado",
        "clear_conversation": "Limpar Conversa",
        
        # Sidebar - About
        "about": "Sobre",
        "tracing": "Rastreamento",
        "tracing_active": "LangSmith Ativo",
        "tracing_disabled": "LangSmith Desativado",
        "project": "Projeto",
        
        # Main Content
        "welcome_title": "Bem-vindo a Lisboa!",
        "welcome_intro": "Sou o seu assistente inteligente para explorar Lisboa, Portugal. Posso ajudar com:",
        "weather_desc": "<strong>Meteorologia</strong> - Condições atuais e previsões",
        "transport_desc": "<strong>Transportes</strong> - Estado do metro, autocarros e comboios",
        "events_desc": "<strong>Eventos</strong> - Eventos culturais e atividades",
        "places_desc": "<strong>Locais</strong> - Pontos de interesse e serviços",
        "planning_desc": "<strong>Planeamento</strong> - Itinerários personalizados",
        "ask_anything": "Pergunte-me qualquer coisa sobre Lisboa!",
        "try_asking": "Experimente perguntar sobre...",
        "chat_placeholder": "Pergunte-me sobre Lisboa...",
        
        # Example Queries (button labels)
        "ex_weather": "Tempo",
        "ex_metro": "Metro",
        "ex_events": "Eventos",
        "ex_services": "Serviços",
        "ex_food": "Gastronomia",
        "ex_planning": "Planeamento",
        
        # Quick Action Queries (full questions in PT)
        "query_weather": "Qual é a previsão do tempo para Lisboa? Inclui avisos meteorológicos ativos.",
        "query_transport": "Qual é o estado atual dos transportes públicos em Lisboa? Inclui Metro, autocarros e comboios.",
        "query_events": "Que eventos culturais estão a acontecer em Lisboa esta semana?",
        "query_attractions": "Quais são as principais atrações turísticas de Lisboa que não posso perder?",
        "query_plan": "Ajuda-me a planear um dia em Lisboa. Estou interessado em história e boa comida.",
        
        # Example Query Texts (full questions in PT)
        "ex_query_weather": "Qual é a previsão do tempo para os próximos 3 dias em Lisboa?",
        "ex_query_metro": "O metro de Lisboa está a funcionar normalmente hoje?",
        "ex_query_events": "Que eventos culturais há este fim de semana em Lisboa?",
        "ex_query_services": "Encontra farmácias e hospitais perto do Rossio",
        "ex_query_food": "Recomenda restaurantes tradicionais portugueses em Alfama",
        "ex_query_planning": "Planeia um itinerário de 2 dias para quem visita Lisboa pela primeira vez",
        
        # Errors
        "error_not_initialized": "Assistente Não Inicializado",
        "error_troubleshooting": "Resolução de Problemas",
        "error_common_issues": "Problemas Comuns:",
        "error_missing_api": "Chave API em Falta",
        "error_local_models": "Modelos Locais (LM Studio / Ollama)",
        "error_network": "Problemas de Rede",
        "retry_init": "Tentar Novamente",
        "error_api_key": "Erro de Chave API (401 Não Autorizado)",
        "error_api_key_msg": "A sua chave API é inválida, expirou ou foi revogada.",
        "error_rate_limit": "Limite de Pedidos Excedido",
        "error_rate_limit_msg": "Excedeu o limite de pedidos da API. Aguarde e tente novamente.",
        "error_connection": "Erro de Ligação",
        "error_connection_msg": "Não foi possível ligar à API. Verifique a sua ligação à internet.",
        "error_generic": "Ocorreu um erro ao processar o seu pedido.",
        "thinking": "A analisar e recolher informação...",
        
        # Footer
        "footer_version": "Assistente Urbano de Lisboa v1.0",
        "footer_made": "André Filipe Gomes Silvestre | Mestrando\nNOVA IMS",
        
        # Info Page
        "info_title": "Sobre Este Assistente",
        "info_objective": "Objetivo",
        "info_objective_text": "Este assistente inteligente foi desenvolvido como parte de uma Tese de Mestrado em Data Science e Advanced Analytics na NOVA IMS (Universidade NOVA de Lisboa). O objetivo é criar uma framework baseada em LLM para planeamento adaptativo de itinerários turísticos e de mobilidade em Lisboa.",
        "info_data_sources": "Fontes de Dados",
        "info_data_sources_text": """O assistente utiliza múltiplas fontes de dados em tempo real e estáticas:

- **API IPMA** - Previsões meteorológicas e avisos
- **Metro de Lisboa** - Estado em tempo real das 4 linhas de metro
- **Carris Metropolitana** - Alertas, paragens e informação de linhas
- **CP (Comboios de Portugal)** - Estado e atrasos de comboios
- **Lisboa Aberta** - Dados abertos (farmácias, hospitais, museus, etc.)
- **VisitLisboa** - Eventos culturais, atrações e pontos de interesse
- **Guia Oficial de Lisboa** - PDF do guia turístico com informação completa""",
        "info_how_to_use": "Como Usar",
        "info_how_to_use_text": """1. **Selecione o seu Fornecedor LLM** - Escolha entre Groq, Google, OpenAI ou modelos locais
2. **Introduza as credenciais** - Forneça a chave API ou URL do servidor
3. **Faça perguntas** - Escreva as suas perguntas em linguagem natural
4. **Use Ações Rápidas** - Clique nos botões da barra lateral para consultas comuns""",
        "info_privacy": "Privacidade e Segurança",
        "info_privacy_text": """- As suas credenciais API são guardadas localmente apenas na sua sessão
- Nenhum dado de conversa é guardado permanentemente
- O rastreamento LangSmith (se ativado) é apenas para fins de desenvolvimento""",
        "info_author": "Autor",
        "info_author_text": """**André Filipe Gomes Silvestre**
Mestrando em Data Science e Advanced Analytics
NOVA IMS - Universidade NOVA de Lisboa
2024/2025""",
    }
}


def t(key: str) -> str:
    """Get translation for current language."""
    lang = st.session_state.get("language", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


# ==========================================================================
# LISBON THEME - CUSTOM CSS
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
}

.lisbon-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: rgba(255,255,255,0.1);
    border-radius: 50%;
    pointer-events: none;
}

.lisbon-header::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 10%;
    width: 150px;
    height: 150px;
    background: rgba(255,255,255,0.08);
    border-radius: 50%;
    pointer-events: none;
}

.lisbon-header h1 {
    color: white;
    margin: 0;
    font-size: 2.4rem;
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    letter-spacing: -0.02em;
    position: relative;
    z-index: 1;
}

.lisbon-header p {
    color: rgba(255,255,255,0.95);
    margin: 0.75rem 0 0 0;
    font-size: 1.15rem;
    font-weight: 400;
    position: relative;
    z-index: 1;
}

/* ============ SIDEBAR ============ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--gray-50) 0%, white 100%);
    border-right: none;
    box-shadow: 4px 0 20px rgba(0,0,0,0.05);
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

section[data-testid="stSidebar"] .stMarkdown h2 {
    color: var(--gray-800);
    font-weight: 600;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: none;
    padding-bottom: 0.5rem;
    margin-bottom: 0.75rem;
}

section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--gray-700);
    font-weight: 600;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

section[data-testid="stSidebar"] button[kind="secondary"] {
    background: white !important;
    border: 1.5px solid var(--gray-200) !important;
    color: var(--gray-700) !important;
}

section[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: var(--lisbon-yellow-light) !important;
    border-color: var(--lisbon-yellow) !important;
    color: var(--gray-800) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(246, 218, 0, 0.2);
}

section[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(135deg, var(--lisbon-orange) 0%, var(--lisbon-orange-light) 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(255, 64, 17, 0.3);
}

section[data-testid="stSidebar"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--lisbon-orange-dark) 0%, var(--lisbon-orange) 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(255, 64, 17, 0.4);
}

/* ============ CHAT MESSAGES ============ */
[data-testid="stChatMessage"] {
    padding: 1.25rem !important;
    margin: 0.75rem 0 !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, var(--lisbon-yellow-light) 0%, white 100%) !important;
    border: 1px solid var(--lisbon-yellow) !important;
    border-radius: 18px 18px 6px 18px !important;
    box-shadow: 0 2px 8px rgba(246, 218, 0, 0.15);
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: white !important;
    border: 1px solid var(--gray-200) !important;
    border-radius: 18px 18px 18px 6px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

/* Chat input */
[data-testid="stChatInput"] > div {
    border-radius: 14px !important;
    border: 2px solid var(--gray-200) !important;
    background: white !important;
    transition: all 0.2s ease;
}

[data-testid="stChatInput"] > div > div > div {
    background: white !important;
}


[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--lisbon-orange) !important;
    box-shadow: 0 0 0 3px rgba(255, 64, 17, 0.1) !important;
}

/* ============ WELCOME CARD ============ */
.welcome-card {
    background: white;
    border: none;
    border-radius: 20px;
    padding: 2.5rem;
    margin: 1.5rem 0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    position: relative;
    overflow: hidden;
}

.welcome-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--lisbon-orange), var(--lisbon-yellow));
}

.welcome-card h3 {
    color: var(--gray-900);
    margin: 0 0 0.5rem 0;
    font-size: 1.75rem;
    font-weight: 700;
}

.welcome-card > p {
    color: var(--gray-600);
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}

.feature-list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0 2rem 0;
}

.feature-item {
    background: var(--gray-50);
    padding: 1rem 1.25rem;
    border-radius: 12px;
    border: none;
    border-left: 3px solid var(--lisbon-yellow);
    transition: all 0.2s ease;
    font-size: 0.95rem;
    color: var(--gray-700);
}

.feature-item:hover {
    background: var(--lisbon-yellow-light);
    border-left-color: var(--lisbon-orange);
    transform: translateX(4px);
}

.feature-item strong {
    color: var(--gray-800);
}

/* ============ EXAMPLE BUTTONS ============ */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.2s ease !important;
}

/* ============ INFO SECTIONS ============ */
.info-section {
    background: white;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin: 1.25rem 0;
    border: none;
    border-left: 4px solid var(--lisbon-orange);
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
}

.info-section h3 {
    color: var(--gray-900);
    margin: 0 0 0.25rem 0;
    font-size: 1.2rem;
    font-weight: 600;
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

.lisbon-footer p {
    margin: 0.3rem 0;
    color: var(--gray-600);
    font-size: 0.875rem;
}

.lisbon-footer p:first-child {
    color: var(--gray-800);
    font-weight: 600;
}

/* ============ METRICS ============ */
[data-testid="stMetric"] {
    background: white;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid var(--gray-200);
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

[data-testid="stMetricValue"] {
    color: var(--lisbon-orange) !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: var(--gray-600) !important;
}

/* ============ DIVIDERS ============ */
hr {
    border: none;
    height: 1px;
    background: var(--gray-200);
    margin: 1.25rem 0;
}

/* ============ ALERTS ============ */
.stSuccess {
    background: linear-gradient(135deg, rgba(14, 224, 113, 0.1) 0%, rgba(14, 224, 113, 0.05) 100%) !important;
    border: 1px solid var(--lisbon-green) !important;
    border-radius: 10px !important;
}

.stWarning {
    background: linear-gradient(135deg, rgba(246, 218, 0, 0.1) 0%, rgba(246, 218, 0, 0.05) 100%) !important;
    border: 1px solid var(--lisbon-yellow) !important;
    border-radius: 10px !important;
}

.stError {
    background: linear-gradient(135deg, rgba(255, 64, 17, 0.1) 0%, rgba(255, 64, 17, 0.05) 100%) !important;
    border: 1px solid var(--lisbon-orange) !important;
    border-radius: 10px !important;
}

/* ============ SELECTBOX ============ */
.stSelectbox > div > div {
    border-radius: 10px !important;
    border-color: var(--gray-200) !important;
}

.stSelectbox > div > div:focus-within {
    border-color: var(--lisbon-orange) !important;
    box-shadow: 0 0 0 2px rgba(255, 64, 17, 0.1) !important;
}

/* ============ TEXT INPUT ============ */
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border-color: var(--gray-200) !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--lisbon-orange) !important;
    box-shadow: 0 0 0 2px rgba(255, 64, 17, 0.1) !important;
}

/* ============ EXPANDER ============ */
.streamlit-expanderHeader {
    background: var(--gray-50) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
}

/* ============ SPINNER ============ */
.stSpinner > div {
    border-top-color: var(--lisbon-orange) !important;
}

/* ============ HIDE STREAMLIT BRANDING ============ */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* ============ SCROLLBAR ============ */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--gray-100);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--gray-300);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--gray-400);
}
</style>
"""


# ==========================================================================
# Page Configuration
# ==========================================================================

st.set_page_config(
    page_title="Lisbon Urban Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Silvestre17/Thesis2025-26_AFGS",
        "Report a bug": "https://github.com/Silvestre17/Thesis2025-26_AFGS/issues",
        "About": """
        # Lisbon Urban Assistant
        
        **Master Thesis Project**  
        Andre Filipe Gomes Silvestre, 2025
        
        An intelligent assistant for tourists and locals in Lisbon.
        """
    }
)


# ==========================================================================
# Session State Initialization
# ==========================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "assistant": None,
        "provider": "lmstudio",
        "initialized": False,
        "error": None,
        "language": "pt",
        "current_page": "chat",
        "credentials": {
            "groq": {"api_key": os.getenv("GROQ_API_KEY", "")},
            "google": {"api_key": os.getenv("GOOGLE_API_KEY", "")},
            "openai": {"api_key": os.getenv("OPENAI_API_KEY", "")},
            "lmstudio": {"base_url": Config.LMSTUDIO_BASE_URL, "model": Config.LMSTUDIO_MODEL_NAME},
            "ollama": {"model": Config.OLLAMA_MODEL_NAME},
        },
        "agent_overrides": {},  # Store custom model selection per agent
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_credentials_env():
    """Set environment variables from stored credentials."""
    creds = st.session_state.credentials
    provider = st.session_state.provider
    
    if provider == "groq" and creds["groq"]["api_key"]:
        os.environ["GROQ_API_KEY"] = creds["groq"]["api_key"]
    elif provider == "google" and creds["google"]["api_key"]:
        os.environ["GOOGLE_API_KEY"] = creds["google"]["api_key"]
    elif provider == "openai" and creds["openai"]["api_key"]:
        os.environ["OPENAI_API_KEY"] = creds["openai"]["api_key"]


@st.cache_resource
def pre_warm_vector_store():
    """
    Pre-warm the vector store to avoid delays during first interaction.
    This is cached globally by Streamlit so it only runs once per server start.
    """
    try:
        initialize_vector_store()
        return True
    except Exception as e:
        print(f"Vector store warming failed: {e}")
        return False


def initialize_assistant(provider: str) -> Tuple[bool, Optional[str]]:
    """Initialize or reinitialize the LisbonAssistant."""
    try:
        set_credentials_env()
        
        # Pre-warm vector store (cached)
        # Only needed if using Multi-Agent or Researcher (which uses tools)
        if Config.USE_MULTI_AGENT:
            with st.spinner("Loading knowledge base (this happens only once)..."):
                pre_warm_vector_store()
        
        # Initialize assistant based on mode
        if Config.USE_MULTI_AGENT:
            # Multi-Agent Mode
            
            # Apply UI overrides if any
            if "agent_overrides" in st.session_state:
                for agent, model_cfg in st.session_state.agent_overrides.items():
                    if agent in Config.AGENT_MODELS:
                        Config.AGENT_MODELS[agent]["model"] = model_cfg
            
            st.session_state.assistant = MultiAgentAssistant()
            
            # =========================================================
            # CONNECTION TEST
            # =========================================================
            # Verify if the configured model is actually reachable
            connection_placeholder = st.empty()
            connection_placeholder.info(f"🔄 Testing connection to supervisor model: {st.session_state.assistant.model_name}...")
            
            try:
                # access the supervisor LLM directly
                test_llm = st.session_state.assistant.supervisor.llm
                # Simple ping
                response = test_llm.invoke("ping")
                # If we get here, connection is successful
                connection_placeholder.success(f"✅ Connection successful! Model is ready.")
                import time
                time.sleep(1.0) # Show success briefly
                connection_placeholder.empty()
                
            except Exception as e:
                connection_placeholder.empty()
                error_msg = f"❌ Connection Timeout/Error: Could not connect to model '{st.session_state.assistant.model_name}'. Check LM Studio server or model name. Details: {str(e)}"
                st.session_state.assistant = None # Rollback
                return False, error_msg

            st.session_state.initialized = True
            st.session_state.provider = provider
            st.session_state.error = None
            return True, None
            
        else:
            # Single-Agent Mode (Legacy)
            st.session_state.assistant = create_assistant(provider)
            st.session_state.initialized = True
            st.session_state.provider = provider
            st.session_state.error = None
            return True, None
    except Exception as e:
        error_msg = str(e)
        st.session_state.error = error_msg
        st.session_state.initialized = False
        # Debug purpose only - uncomment to see full traceback
        # traceback.print_exc()
        return False, error_msg


# ==========================================================================
# UI Components
# ==========================================================================

def render_header():
    """Render the Lisbon-themed header."""
    st.markdown(f"""
    <div class="lisbon-header">
        <h1>🏛️ {t("app_title")}</h1>
        <p>{t("app_subtitle")}</p>
    </div>
    """, unsafe_allow_html=True)


def render_language_selector():
    """Render language selector in sidebar."""
    languages = {"🇬🇧 English": "en", "🇵🇹 Português": "pt"}
    current_lang = st.session_state.language
    
    selected = st.selectbox(
        t("language"),
        options=list(languages.keys()),
        index=list(languages.values()).index(current_lang),
        key="lang_selector"
    )
    
    if languages[selected] != current_lang:
        st.session_state.language = languages[selected]
        st.rerun()


def render_provider_credentials():
    """Render provider selection and credentials input."""
    st.markdown(f"### {t('llm_provider')}")
    
    provider_info = {
        "lmstudio": ("LM Studio", "Local server", "local"),
        "ollama": ("Ollama", "Local Ollama models", "ollama"),
        "groq": ("Groq", "Fast inference with Qwen/Llama", "api_key"),
        "google": ("Google Gemini", "Google's Gemini 3 models", "api_key"),
        "openai": ("OpenAI", "GPT-5 models", "api_key"),
    }
    
    provider_names = [info[0] for info in provider_info.values()]
    provider_keys = list(provider_info.keys())
    
    current_idx = provider_keys.index(st.session_state.provider) if st.session_state.provider in provider_keys else 0
    
    selected_display = st.selectbox(
        t("select_provider"),
        options=provider_names,
        index=current_idx,
        key="provider_select"
    )
    
    selected_provider = provider_keys[provider_names.index(selected_display)]
    provider_type = provider_info[selected_provider][2]
    
    st.caption(provider_info[selected_provider][1])
    st.markdown(f"#### {t('api_credentials')}")
    
    credentials_changed = False
    
    if provider_type == "api_key":
        api_key = st.text_input(
            t("api_key"),
            value=st.session_state.credentials[selected_provider].get("api_key", ""),
            type="password",
            placeholder=t("api_key_placeholder"),
            key=f"api_key_{selected_provider}"
        )
        if api_key != st.session_state.credentials[selected_provider].get("api_key", ""):
            st.session_state.credentials[selected_provider]["api_key"] = api_key
            credentials_changed = True
            
    elif provider_type == "local":
        # LM Studio: Server URL
        base_url = st.text_input(
            t("local_url"),
            value=st.session_state.credentials["lmstudio"].get("base_url", Config.LMSTUDIO_BASE_URL),
            placeholder=t("local_url_placeholder"),
            key="lmstudio_url"
        )
        # LM Studio: Model name on separate line for better visibility
        model = st.text_input(
            t("model_name"),
            value=st.session_state.credentials["lmstudio"].get("model", Config.LMSTUDIO_MODEL_NAME),
            placeholder=Config.LMSTUDIO_MODEL_NAME,
            key="lmstudio_model",
            help="Nome do modelo carregado no LM Studio"
        )
        if (base_url != st.session_state.credentials["lmstudio"].get("base_url", "") or 
            model != st.session_state.credentials["lmstudio"].get("model", "")):
            st.session_state.credentials["lmstudio"]["base_url"] = base_url
            st.session_state.credentials["lmstudio"]["model"] = model
            credentials_changed = True
            
    elif provider_type == "ollama":
        model = st.text_input(
            t("model_name"),
            value=st.session_state.credentials["ollama"].get("model", "llama3.2"),
            placeholder=t("model_name_placeholder"),
            key="ollama_model"
        )
        if model != st.session_state.credentials["ollama"]["model"]:
            st.session_state.credentials["ollama"]["model"] = model
            credentials_changed = True

    # =========================================================================
    # ADVANCED AGENT CONFIGURATION (Multi-Agent Only)
    # =========================================================================
    if Config.USE_MULTI_AGENT:
        with st.expander("🛠️ Advanced: Agent Models"):
            st.caption("Customize models for each agent. Default: Config.py")
            
            # Agents list
            agents = ["supervisor", "weather", "transport", "researcher", "planner"]
            
            for agent in agents:
                # Get current config or default
                default_model = Config.AGENT_MODELS.get(agent, {}).get("model", Config.LMSTUDIO_MODEL_NAME)
                current_override = st.session_state.agent_overrides.get(agent, default_model)
                
                # Render input for this agent
                new_model = st.text_input(
                    f"{agent.capitalize()} Model",
                    value=current_override,
                    key=f"agent_model_{agent}",
                    help=f"Model for {agent} agent"
                )
                
                # Check for changes
                if new_model != current_override:
                    st.session_state.agent_overrides[agent] = new_model
                    credentials_changed = True
    
    needs_reinit = (selected_provider != st.session_state.provider or 
                   not st.session_state.initialized or credentials_changed)
    
    if needs_reinit:
        if st.button(t('save_credentials'), use_container_width=True, type="primary"):
            with st.spinner("Connecting..."):
                st.session_state.provider = selected_provider
                success, error = initialize_assistant(selected_provider)
                if success:
                    st.success(t('assistant_ready'))
                    st.rerun()
                else:
                    st.error(f"{t('initialization_failed')}: {error}")
    else:
        st.success(t('assistant_ready'))
    
    return selected_provider


def render_quick_actions() -> Optional[str]:
    """Render quick action buttons."""
    st.markdown(f"## {t('quick_actions')}")
    
    actions = [
        ("🌤️", t("weather_summary"), t("query_weather")),
        ("🚇", t("transport_status"), t("query_transport")),
        ("🎭", t("upcoming_events"), t("query_events")),
        ("📍", t("top_attractions"), t("query_attractions")),
        ("🗺️", t("plan_my_day"), t("query_plan")),
    ]
    
    for icon, label, query in actions:
        if st.button(f"{icon} {label}", use_container_width=True, key=f"qa_{label}"):
            return query
    return None


def render_session_info():
    """Render session information."""
    st.markdown(f"## {t('session_info')}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(t("messages"), len(st.session_state.messages))
    with col2:
        status = "🟢" if st.session_state.initialized else "🔴"
        st.metric(t("status"), status)
    
    if st.session_state.initialized and st.session_state.assistant:
        st.caption(f"Model: {st.session_state.assistant.model_name}")
    
    if st.button(f"🗑️ {t('clear_conversation')}", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.assistant:
            st.session_state.assistant.reset()
        st.rerun()


def render_about_section():
    """Render about section in sidebar."""
    st.markdown(f"## {t('about')}")
    st.markdown("""**Master Thesis Project**  
NOVA IMS, 2025

*LLM-Powered Urban Exploration*""")
    
    learn_more_text = "Saber Mais" if st.session_state.language == "pt" else "Learn More"
    if st.button(f"📖 {learn_more_text}", use_container_width=True, key="info_btn"):
        st.session_state.current_page = "info"
        st.rerun()
    
    st.markdown("[🔗 GitHub](https://github.com/Silvestre17/Thesis2025-26_AFGS)")


def render_tracing_info():
    """Render LangSmith tracing information."""
    st.markdown(f"## {t('tracing')}")
    
    langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    langsmith_project = os.getenv("LANGCHAIN_PROJECT", "default")
    
    if langsmith_enabled:
        st.success(t('tracing_active'))
        st.caption(f"{t('project')}: {langsmith_project}")
    else:
        st.warning(t('tracing_disabled'))


def render_sidebar() -> Tuple[str, Optional[str]]:
    """Render complete sidebar."""
    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Chat", use_container_width=True, 
                        type="primary" if st.session_state.current_page == "chat" else "secondary"):
                st.session_state.current_page = "chat"
                st.rerun()
        with col2:
            if st.button("Info", use_container_width=True,
                        type="primary" if st.session_state.current_page == "info" else "secondary"):
                st.session_state.current_page = "info"
                st.rerun()
        
        st.divider()
        st.markdown(f"## {t('settings')}")
        render_language_selector()
        st.divider()
        selected_provider = render_provider_credentials()
        st.divider()
        quick_action = render_quick_actions()
        st.divider()
        render_session_info()
        st.divider()
        render_about_section()
        st.divider()
        render_tracing_info()
    
    return selected_provider, quick_action


def render_info_page():
    """Render the information/about page."""
    st.markdown(f"# {t('info_title')}")
    
    st.markdown(f"""<div class="info-section"><h3>{t('info_objective')}</h3></div>""", unsafe_allow_html=True)
    st.markdown(t('info_objective_text'))
    
    st.markdown(f"""<div class="info-section"><h3>{t('info_data_sources')}</h3></div>""", unsafe_allow_html=True)
    st.markdown(t('info_data_sources_text'))
    
    st.markdown(f"""<div class="info-section"><h3>{t('info_how_to_use')}</h3></div>""", unsafe_allow_html=True)
    st.markdown(t('info_how_to_use_text'))
    
    st.markdown(f"""<div class="info-section"><h3>{t('info_privacy')}</h3></div>""", unsafe_allow_html=True)
    st.markdown(t('info_privacy_text'))
    
    st.markdown(f"""<div class="info-section"><h3>{t('info_author')}</h3></div>""", unsafe_allow_html=True)
    st.markdown(t('info_author_text'))
    
    back_text = "Voltar ao Chat" if st.session_state.language == "pt" else "Back to Chat"
    if st.button(f"💬 {back_text}", type="primary", use_container_width=True):
        st.session_state.current_page = "chat"
        st.rerun()


def render_chat_messages():
    """Render chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)


def render_example_queries() -> Optional[str]:
    """Render example query buttons."""
    st.markdown(f"### {t('try_asking')}")
    
    examples = [
        ("🌤️", t("ex_weather"), t("ex_query_weather")),
        ("🚇", t("ex_metro"), t("ex_query_metro")),
        ("🎭", t("ex_events"), t("ex_query_events")),
        ("🏥", t("ex_services"), t("ex_query_services")),
        ("🍽️", t("ex_food"), t("ex_query_food")),
        ("🗺️", t("ex_planning"), t("ex_query_planning")),
    ]
    
    cols = st.columns(3)
    selected = None
    
    for i, (icon, label, query) in enumerate(examples):
        with cols[i % 3]:
            if st.button(f"{icon} {label}", key=f"ex_{i}", use_container_width=True):
                selected = query
    
    return selected


# def render_error_panel():
#     """Render error panel when initialization fails."""
#     st.error(t('error_not_initialized'))
    
#     with st.expander(t('error_troubleshooting'), expanded=True):
#         st.markdown(f"""
# **{t('error_common_issues')}**

# 1. **{t('error_missing_api')}**
#    - Groq: Get key from [console.groq.com](https://console.groq.com/keys)
#    - Google: Get key from [Google AI Studio](https://aistudio.google.com/app/apikey)
#    - OpenAI: Get key from [platform.openai.com](https://platform.openai.com/api-keys)

# 2. **{t('error_local_models')}**
#    - LM Studio: Start server on port 1234
#    - Ollama: Run 'ollama serve' and ensure model is downloaded

# 3. **{t('error_network')}**
#    - Check internet connection
#    - Verify firewall settings
#         """)
        
#         if st.session_state.error:
#             # Debug purpose only - show full error
#             st.code(st.session_state.error, language="text")
    
#     if st.button(t('retry_init'), use_container_width=True, type="primary"):
#         with st.spinner("..."):
#             success, _ = initialize_assistant(st.session_state.provider)
#             if success:
#                 st.rerun()


def process_user_input(user_input: str):
    """Process user input and generate response."""
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        try:
            # Dynamic Status Update Implementation
            with st.status("🤔 A analisar e recolher informação...", expanded=False) as status:
                def update_ui_status(message: str):
                    """Callback to update UI status from agent graph."""
                    status.update(label=message, state="running")
                    
                try:
                    # Enable verbose mode and pass status callback
                    response = st.session_state.assistant.chat(
                        user_input, 
                        verbose=True, 
                        on_status_change=update_ui_status
                    )
                    
                    # Mark as complete
                    status.update(label="✅ Resposta pronta!", state="complete", expanded=False)
                    
                except Exception as e:
                    status.update(label="❌ Erro no processamento", state="error", expanded=True)
                    raise e # Re-raise to be caught by the outer except block

            # Display response only if successful (outside status container)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
                error_str = str(e).lower()
                
                if "401" in error_str or "unauthorized" in error_str:
                    error_msg = f"{t('error_api_key')}\n\n{t('error_api_key_msg')}"
                elif "rate" in error_str or "limit" in error_str:
                    error_msg = f"{t('error_rate_limit')}\n\n{t('error_rate_limit_msg')}"
                elif "timeout" in error_str or "connection" in error_str:
                    error_msg = f"{t('error_connection')}\n\n{t('error_connection_msg')}"
                else:
                    error_msg = f"{t('error_generic')}\n\n{str(e)}"
                
                # Format full error message with traceback
                full_error_content = f"""
### ⚠️ Error
{error_msg}

<details>
<summary>Technical Details</summary>

```python
{traceback.format_exc()}
```
</details>
"""
                st.markdown(full_error_content, unsafe_allow_html=True)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_error_content
                })


def render_footer():
    """Render the application footer."""
    st.markdown(f"""
    <div class="lisbon-footer">
        <p>{t('footer_version')}</p>
        <p>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <p>{t('footer_made')}</p>
    </div>
    """, unsafe_allow_html=True)


def render_welcome_section():
    """Render welcome section for new users."""
    st.markdown(f"""
    <div class="welcome-card">
        <h3>{t('welcome_title')}</h3>
        <p>{t('welcome_intro')}</p>
        <div class="feature-list">
            <div class="feature-item">{t('weather_desc')}</div>
            <div class="feature-item">{t('transport_desc')}</div>
            <div class="feature-item">{t('events_desc')}</div>
            <div class="feature-item">{t('places_desc')}</div>
            <div class="feature-item">{t('planning_desc')}</div>
        </div>
        <p><strong>{t('ask_anything')}</strong></p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================================
# Main Application
# ==========================================================================

def main():
    """Main application entry point."""
    st.markdown(LISBON_CSS, unsafe_allow_html=True)
    initialize_session_state()
    render_header()
    
    selected_provider, quick_action = render_sidebar()
    
    if st.session_state.current_page == "info":
        render_info_page()
        render_footer()
        return
    
    main_container = st.container()
    
    with main_container:
        # Initialize assistant if not already done or if provider changed
        if not st.session_state.initialized or st.session_state.provider != selected_provider:
            with st.spinner("Starting Lisbon Urban Assistant..."):
                success, error = initialize_assistant(selected_provider)
                if not success:
                    st.error(f"Failed to initialize assistant: {error}")
                    st.info("Please check your API credentials in the sidebar.")
                    render_footer()
                    return
        
        # Safety check: ensure assistant exists
        if not st.session_state.assistant:
            st.error("Assistant not initialized. Please refresh the page.")
            render_footer()
            return
        
        render_chat_messages()
        
        example_query = None
        if not st.session_state.messages:
            render_welcome_section()
            example_query = render_example_queries()
        
        if quick_action:
            process_user_input(quick_action)
            st.rerun()
        
        if example_query:
            process_user_input(example_query)
            st.rerun()
    
    if user_input := st.chat_input(t("chat_placeholder"), key="chat_input"):
        process_user_input(user_input)
        st.rerun()
    
    render_footer()


# ==========================================================================
# Entry Point
# ==========================================================================

if __name__ == "__main__":
    main()
