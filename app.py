# ==========================================================================
# Master Thesis - Streamlit Application
#   - André Filipe Gomes Silvestre, 20240502
#
#   Streamlit user interface for LISBOA, the Lisbon Itinerary System Based On
#   AI. Provides the interactive chat workflow, provider selection, quick
#   actions, tracing controls, debug visibility, and response rendering for the
#   multi-agent runtime.
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
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote

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
except Exception as exc:
    logging.getLogger(__name__).debug("Streamlit torch path workaround skipped: %s", exc)

sys.path.insert(0, ".")

from agent.utils.startup_resources import run_startup_preload as _run_startup_preload_impl
from config import Config

PORTUGAL_FLAG_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 20">
  <defs>
    <clipPath id="pt-flag-clip">
      <rect width="30" height="20" rx="1.2"/>
    </clipPath>
  </defs>
  <g clip-path="url(#pt-flag-clip)">
    <rect width="30" height="20" fill="#ff0000"/>
    <rect width="12" height="20" fill="#006600"/>
  </g>
  <g transform="translate(12 10)">
    <g fill="none" stroke="#1f1f1f" stroke-linecap="round" stroke-linejoin="round" stroke-width=".32">
      <circle r="3.95"/>
      <ellipse rx="3.95" ry="1.18"/>
      <ellipse rx="1.18" ry="3.95"/>
      <path d="M-3.38-2.1c1.56.82 4.9 2.86 6.76 4.2M3.38-2.1c-1.56.82-4.9 2.86-6.76 4.2"/>
    </g>
    <g fill="none" stroke="#fff200" stroke-linecap="round" stroke-linejoin="round" stroke-width=".76">
      <circle r="3.95"/>
      <ellipse rx="3.95" ry="1.18"/>
      <ellipse rx="1.18" ry="3.95"/>
      <path d="M-3.38-2.1c1.56.82 4.9 2.86 6.76 4.2M3.38-2.1c-1.56.82-4.9 2.86-6.76 4.2"/>
    </g>
    <path d="M-2.16-2.45h4.32v3.72c0 1.52-.94 2.72-2.16 3.16-1.22-.44-2.16-1.64-2.16-3.16z" fill="#f00000" stroke="#fff" stroke-width=".5"/>
    <path d="M-1.08-1.28h2.16v2.38c0 .96-.45 1.68-1.08 1.96-.63-.28-1.08-1-1.08-1.96z" fill="#fff" stroke="#1f1f1f" stroke-width=".18"/>
    <g fill="#0032a0">
      <circle cx="0" cy="-.78" r=".22"/>
      <circle cx="-.56" cy=".02" r=".22"/>
      <circle cx=".56" cy=".02" r=".22"/>
      <circle cx="0" cy=".78" r=".22"/>
      <circle cx="0" cy="1.48" r=".22"/>
    </g>
    <g fill="#fff">
      <circle cx="0" cy="-.78" r=".06"/>
      <circle cx="-.56" cy=".02" r=".06"/>
      <circle cx=".56" cy=".02" r=".06"/>
      <circle cx="0" cy=".78" r=".06"/>
      <circle cx="0" cy="1.48" r=".06"/>
    </g>
  </g>
</svg>
""".strip()

UK_FLAG_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 20">
  <rect width="30" height="20" fill="#012169"/>
  <path d="M0 0l30 20M30 0L0 20" stroke="#fff" stroke-width="4"/>
  <path d="M0 0l30 20M30 0L0 20" stroke="#c8102e" stroke-width="2"/>
  <path d="M15 0v20M0 10h30" stroke="#fff" stroke-width="6"/>
  <path d="M15 0v20M0 10h30" stroke="#c8102e" stroke-width="3.4"/>
</svg>
""".strip()

FLAG_IMAGE_URIS = {
    "pt": f"data:image/svg+xml;charset=utf-8,{quote(PORTUGAL_FLAG_SVG)}",
    "en": f"data:image/svg+xml;charset=utf-8,{quote(UK_FLAG_SVG)}",
}


LANGUAGE_OPTIONS = {
    "pt": {"label": "Português", "short": "PT"},
    "en": {"label": "English", "short": "EN"},
}

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
        "info_title": "LISBOA",
        "info_subtitle": "Lisbon Itinerary System Based On AI",
        "info_intro": "An academic multi-agent prototype that helps users ask grounded questions about Lisbon tourism, mobility, weather, local services, and itineraries.",
        "info_badge": "Master's Thesis Project",
        "info_stat_agents_value": "6",
        "info_stat_agents_label": "Coordinated Agents",
        "info_stat_tools_value": "45",
        "info_stat_tools_label": "Grounded Tools",
        "info_stat_scope_value": "PT/EN",
        "info_stat_scope_label": "Bilingual Interface",
        "info_f1_title": "Tourism and Culture",
        "info_f1_desc": "Attractions, monuments, events, neighbourhoods, restaurants, and local context anchored in Lisbon-focused sources.",
        "info_f2_title": "Mobility",
        "info_f2_desc": "Public-transport guidance across Metro, Carris Urban, Carris Metropolitana, CP, and multimodal route logic.",
        "info_f3_title": "Weather",
        "info_f3_desc": "IPMA forecasts and active warnings used to adapt visit suggestions, clothing advice, and daily plans.",
        "info_f4_title": "Plans and Services",
        "info_f4_desc": "Itineraries, nearby services, useful municipal information, and Lisbon-specific practical guidance.",
        "info_architecture_title": "System Architecture",
        "info_architecture_desc": "A *supervisor-worker* architecture routes each request, retrieves evidence through specialised tools, validates the answer, and only synthesises an itinerary when planning is needed.",
        "info_flow_title": "How LISBOA produces an answer",
        "info_flow_1_title": "Data Sources",
        "info_flow_1_desc": "Tourism, weather, transport, open-data, and local-knowledge sources provide the evidence base.",
        "info_flow_2_title": "Tools and Context",
        "info_flow_2_desc": "Retrieval, geospatial search, weather, and mobility tools convert source data into usable context.",
        "info_flow_3_title": "Agent Coordination",
        "info_flow_3_desc": "The supervisor selects the right worker agents and preserves conversation context across follow-up questions.",
        "info_flow_4_title": "Final Response",
        "info_flow_4_desc": "The quality layer checks completeness, sources, scope, and formatting before the response is shown.",
        "info_audience_title": "Designed for Tourists and Residents",
        "info_audience_desc": "The same system supports tourist planning and everyday urban decisions; it changes the evidence it prioritises and the response format it produces.",
        "info_tourists_title": "Tourists",
        "info_tourists_desc": "Useful for discovering monuments, events, neighbourhoods, restaurants, and weather-aware itineraries.",
        "info_residents_title": "Residents",
        "info_residents_desc": "Useful for mobility decisions, nearby services, local events, municipal information, and practical planning.",
        "info_framework_title": "Technical Map",
        "info_framework_desc": "The diagram summarises the implemented data sources, tool layer, agent orchestration, interface, and evaluation.",
        "info_source_visitlisboa_desc": "Tourism, events, and cultural knowledge",
        "info_source_ipma_desc": "Official forecasts and weather warnings",
        "info_source_metro_desc": "Metro service status and operational information",
        "info_source_carris_desc": "Lisbon buses and trams",
        "info_source_cm_desc": "Metropolitan bus network",
        "info_source_cp_desc": "Railway stations and service data",
        "info_source_lisboa_aberta_desc": "Municipal open data and services",
        "back_to_chat": "Back to Chat",
        "feat_atmosfera": "🌤️ Atmosphere",
        "feat_mobilidade": "🚇 Mobility",
        "feat_cultura": "🎭 Culture",
        "feat_mapa": "📍 Places",
        "feat_roteiros": "🗺️ Itineraries",
        "info_scope_title": "Scope and Limits",
        "info_scope_items": [
            "It is prepared for questions within Lisbon or the Lisbon Metropolitan Area using the integrated data sources.",
            "Live details depend on the availability and freshness of the connected public sources.",
            "It does not replace official operators, emergency services, booking platforms, or payment systems.",
            "When a request is outside the supported scope or asks for information the system cannot verify, LISBOA should state the limitation clearly and suggest a safer next step.",
        ],
        "info_examples_title": "What You Can Ask",
        "info_examples_desc": "These examples mirror the user-study tasks and show the intended scope of the prototype.",
        "info_example_weather_title": "Weather-Aware Questions",
        "info_example_weather_items": [
            "What is the weather like in Lisbon today? Should I take a jacket?",
            "What is the weather forecast for the next 3 days in Lisbon?",
        ],
        "info_example_mobility_title": "Mobility Questions",
        "info_example_mobility_items": [
            "How do I get from Rossio to Belém by public transport?",
            "What is the next train from Entrecampos to Sete-Rios?",
        ],
        "info_example_tourism_title": "Tourism and Local Information",
        "info_example_tourism_items": [
            "What are three well-known monuments in Belém?",
            "Is there a pharmacy near Rossio?",
        ],
        "info_example_plan_title": "Itinerary Planning",
        "info_example_plan_items": [
            "Plan an afternoon in Belém from Chiado, with historical context, realistic public transport, and one custard tart stop.",
        ],
        "info_data_sources": "Integrated Sources",
        "info_author": "Author",
        "info_author_project": "LISBOA: Lisbon Itinerary System Based On AI",
        "info_author_role": "Master's Student",
        "info_author_degree": "Data Science and Advanced Analytics",
        "info_author_affiliation": "NOVA IMS - Universidade NOVA de Lisboa",
        "info_author_year": "Academic year 2025/2026",
        "info_author_github": "GitHub",
        "info_author_linkedin": "LinkedIn",
        "discover_eyebrow": "New here?",
        "discover_title": "Discover how LISBOA works",
        "discover_subtitle": "A multi-agent system grounding live transport, weather, events and tourism data into feasible itineraries for Lisbon.",
        "discover_cta": "Explore the system",
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
        "query_plan": "Cria um roteiro otimizado de 1 dia com monumentos históricos e gastronomia tradicional.",
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
        "info_title": "LISBOA",
        "info_subtitle": "Lisbon Itinerary System Based On AI",
        "info_intro": "Um protótipo académico multi-agente que ajuda a fazer perguntas fundamentadas sobre turismo, mobilidade, meteorologia, serviços locais e roteiros em Lisboa.",
        "info_badge": "Projeto de Tese de Mestrado",
        "info_stat_agents_value": "6",
        "info_stat_agents_label": "Agentes Coordenados",
        "info_stat_tools_value": "45",
        "info_stat_tools_label": "Ferramentas Especializadas",
        "info_stat_scope_value": "PT/EN",
        "info_stat_scope_label": "Interface Bilingue",
        "info_f1_title": "Turismo e Cultura",
        "info_f1_desc": "Atrações, monumentos, eventos, bairros, restaurantes e contexto local ancorados em fontes centradas em Lisboa.",
        "info_f2_title": "Mobilidade",
        "info_f2_desc": "Apoio a deslocações em Metro, Carris Urban, Carris Metropolitana, CP e lógica multimodal.",
        "info_f3_title": "Meteorologia",
        "info_f3_desc": "Previsões e avisos do IPMA usados para adaptar visitas, roupa recomendada e planos do dia.",
        "info_f4_title": "Planos e Serviços",
        "info_f4_desc": "Roteiros, serviços próximos, informação municipal útil e apoio prático específico de Lisboa.",
        "info_architecture_title": "Arquitetura do Sistema",
        "info_architecture_desc": "Uma arquitetura *supervisor-worker* encaminha cada pedido, recolhe evidência com ferramentas especializadas, valida a resposta e só sintetiza itinerários quando há planeamento.",
        "info_flow_title": "Como o LISBOA produz uma resposta",
        "info_flow_1_title": "Fontes de Dados",
        "info_flow_1_desc": "Fontes de turismo, meteorologia, transportes, dados abertos e conhecimento local fornecem a base factual.",
        "info_flow_2_title": "Ferramentas e Contexto",
        "info_flow_2_desc": "Retrieval, pesquisa geoespacial, meteorologia e mobilidade transformam dados de origem em contexto utilizável.",
        "info_flow_3_title": "Coordenação por Agentes",
        "info_flow_3_desc": "O supervisor escolhe os agentes certos e preserva o contexto da conversa em perguntas de seguimento.",
        "info_flow_4_title": "Resposta Final",
        "info_flow_4_desc": "A camada de qualidade verifica completude, fontes, âmbito e formatação antes de mostrar a resposta.",
        "info_audience_title": "Pensado para Turistas e Residentes",
        "info_audience_desc": "O mesmo sistema apoia planeamento turístico e decisões urbanas do dia a dia; muda a evidência que prioriza e o formato de resposta que produz.",
        "info_tourists_title": "Turistas",
        "info_tourists_desc": "Útil para descobrir monumentos, eventos, bairros, restaurantes e roteiros adaptados ao tempo.",
        "info_residents_title": "Residentes",
        "info_residents_desc": "Útil para mobilidade, serviços próximos, eventos locais, informação municipal e decisões práticas.",
        "info_framework_title": "Mapa Técnico",
        "info_framework_desc": "O diagrama resume as fontes de dados implementadas, a camada de ferramentas, a orquestração por agentes, a interface e a avaliação.",
        "info_source_visitlisboa_desc": "Turismo, eventos e conhecimento cultural",
        "info_source_ipma_desc": "Previsões e avisos meteorológicos oficiais",
        "info_source_metro_desc": "Estado do serviço e informação operacional do Metro",
        "info_source_carris_desc": "Autocarros e elétricos de Lisboa",
        "info_source_cm_desc": "Rede metropolitana de autocarros",
        "info_source_cp_desc": "Estações e dados de serviço ferroviário",
        "info_source_lisboa_aberta_desc": "Dados abertos e serviços municipais",
        "back_to_chat": "Voltar ao Chat",
        "feat_atmosfera": "🌤️ Atmosfera",
        "feat_mobilidade": "🚇 Mobilidade",
        "feat_cultura": "🎭 Cultura",
        "feat_mapa": "📍 Locais",
        "feat_roteiros": "🗺️ Roteiros",
        "info_scope_title": "Âmbito e Limites",
        "info_scope_items": [
            "Está preparado para perguntas dentro de Lisboa ou da Área Metropolitana de Lisboa com as fontes de dados integradas.",
            "Os detalhes em tempo real dependem da disponibilidade e atualização das fontes públicas ligadas ao sistema.",
            "Não substitui operadores oficiais, serviços de emergência, plataformas de reserva ou sistemas de pagamento.",
            "Quando o pedido sai do âmbito suportado ou exige informação que o sistema não consegue verificar, o LISBOA deve explicitar a limitação e orientar para uma alternativa mais segura.",
        ],
        "info_examples_title": "O Que Pode Perguntar",
        "info_examples_desc": "Estes exemplos seguem as tarefas do estudo com utilizadores e mostram o âmbito previsto do protótipo.",
        "info_example_weather_title": "Perguntas Meteorológicas",
        "info_example_weather_items": [
            "Como está o tempo hoje em Lisboa? Devo levar casaco?",
            "Qual é a previsão do tempo para os próximos 3 dias em Lisboa?",
        ],
        "info_example_mobility_title": "Perguntas de Mobilidade",
        "info_example_mobility_items": [
            "Como vou do Rossio a Belém de transporte público?",
            "Qual é o próximo comboio de Entrecampos para Sete-Rios?",
        ],
        "info_example_tourism_title": "Turismo e Informação Local",
        "info_example_tourism_items": [
            "Quais são três monumentos conhecidos em Belém?",
            "Há alguma farmácia perto do Rossio?",
        ],
        "info_example_plan_title": "Planeamento de Roteiros",
        "info_example_plan_items": [
            "Planeia uma tarde em Belém a partir do Chiado, com contexto histórico, transporte público realista e uma paragem para pastel de nata.",
        ],
        "info_data_sources": "Fontes Integradas",
        "info_author": "Autor",
        "info_author_project": "LISBOA: Lisbon Itinerary System Based On AI",
        "info_author_role": "Mestrando",
        "info_author_degree": "Data Science and Advanced Analytics",
        "info_author_affiliation": "NOVA IMS - Universidade NOVA de Lisboa",
        "info_author_year": "Ano letivo 2025/2026",
        "info_author_github": "GitHub",
        "info_author_linkedin": "LinkedIn",
        "discover_eyebrow": "Primeira visita?",
        "discover_title": "Descubra como o LISBOA funciona",
        "discover_subtitle": "Um sistema multi-agente que combina dados ao vivo de transportes, meteorologia, eventos e turismo em roteiros viáveis para Lisboa.",
        "discover_cta": "Explorar o sistema",
    },
}


def t(key: str) -> str:
    lang = st.session_state.get("language", "pt")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


def t_list(key: str) -> list[str]:
    """Return a translated list value for structured UI sections."""
    lang = st.session_state.get("language", "pt")
    value = TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, [])
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if value:
        return [str(value)]
    return []


def md_to_html(text: str) -> str:
    """Convert minimal inline markdown to safe HTML for use in rendered blocks."""
    html_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", html_text)


def render_html_block(content: str) -> None:
    """Render raw HTML reliably, preferring `st.html()` when available."""
    if hasattr(st, "html"):
        st.html(content)
        return
    st.markdown(content, unsafe_allow_html=True)


def build_info_feature_card_html(
    icon: str,
    title: str,
    description: str,
    tone: str,
) -> str:
    """Build a feature card used on the Info page."""
    safe_tone = re.sub(r"[^a-z0-9_-]", "", tone.lower()) or "red"
    return (
        f'<article class="info-card info-card-{safe_tone}">'
        f'<div class="info-card-icon">{html.escape(icon)}</div>'
        f'<div class="info-card-title">{html.escape(title)}</div>'
        f'<div class="info-card-desc">{html.escape(description)}</div>'
        '</article>'
    )


def build_info_detail_card_html(
    icon: str,
    title: str,
    body: str,
    tone: str,
    extra_class: str = "",
) -> str:
    """Build a detail card used on the Info page."""
    safe_tone = re.sub(r"[^a-z0-9_-]", "", tone.lower()) or "red"
    safe_extra = re.sub(r"[^a-z0-9_ -]", "", extra_class).strip()
    extra = f" {safe_extra}" if safe_extra else ""
    return (
        f'<article class="info-detail-card info-detail-{safe_tone}{extra}">'
        f'<div class="info-detail-title">{html.escape(icon)} <span>{html.escape(title)}</span></div>'
        f'<div class="info-detail-body">{body}</div>'
        '</article>'
    )


def build_info_list_html(items: list[str]) -> str:
    """Build a safe HTML list for Info page detail blocks."""
    if not items:
        return ""
    return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"


def build_info_example_card_html(icon: str, title: str, examples: list[str], tone: str) -> str:
    """Build a prompt-example card used on the Info page."""
    safe_tone = re.sub(r"[^a-z0-9_-]", "", tone.lower()) or "red"
    prompt_items = "".join(f"<li>{html.escape(example)}</li>" for example in examples)
    return (
        f'<article class="info-example-card info-example-{safe_tone}">'
        f'<div class="info-example-title">{html.escape(icon)} <span>{html.escape(title)}</span></div>'
        f'<ul>{prompt_items}</ul>'
        '</article>'
    )


def build_info_source_link_html(label: str, description: str, url: str) -> str:
    """Build a compact external source link used on the Info page."""
    return (
        '<a class="info-source-link" '
        f'href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">'
        '<span>'
        f'<strong>{html.escape(label)}</strong>'
        f'<small>{html.escape(description)}</small>'
        '</span>'
        '<span aria-hidden="true">↗</span>'
        '</a>'
    )


# ==========================================================================
# PRODUCTION UI - CUSTOM CSS AND ASSETS
# ==========================================================================


@st.cache_data(show_spinner=False, max_entries=16)
def get_base64_image(image_path: str, modified_ns: int = 0) -> str:
    """Return a base64-encoded image file for inline UI assets."""
    _ = modified_ns
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except OSError:
        return ""


def build_image_data_uri(image_path: str) -> str:
    """Build a browser-ready data URI for a local image asset."""
    try:
        modified_ns = os.stat(image_path).st_mtime_ns if image_path else 0
    except OSError:
        modified_ns = 0

    image_b64 = get_base64_image(image_path, modified_ns)
    if not image_b64:
        return ""

    suffix = os.path.splitext(image_path)[1].lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }.get(suffix, "image/png")
    return f"data:{mime_type};base64,{image_b64}"


def resolve_asset_path(*parts: str) -> str:
    """Return an absolute asset path only when the file exists."""
    asset_path = os.path.join(os.path.dirname(__file__), "img", *parts)
    if os.path.exists(asset_path):
        return asset_path

    logging.warning("UI asset missing: %s", asset_path)
    return ""


def resolve_first_asset_path(*filenames: str) -> str:
    """Return the first existing asset path from the provided filenames."""
    for filename in filenames:
        asset_path = os.path.join(os.path.dirname(__file__), "img", filename)
        if os.path.exists(asset_path):
            return asset_path

    logging.warning("No UI asset candidate exists: %s", ", ".join(filenames))
    return ""


# Auto load assets
banner_path = resolve_asset_path("BannerLSIBOA_21-9_optimized.webp")
logo_path = resolve_first_asset_path(
    "Logo_1-1_WithoutBG_optimized.webp",
    "Logo_1-1_WithoutBG.png",
)
small_logo_path = resolve_asset_path("t.png")
framework_path = resolve_asset_path("LISBOA_Framework.svg")

banner_url = build_image_data_uri(banner_path)
logo_url = build_image_data_uri(logo_path)
framework_url = build_image_data_uri(framework_path)

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
    --shadow-md: 0 18px 44px rgba(148, 148, 148, 0.16);
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

/* Language selector: avoids OS emoji flag rendering and Streamlit widget truncation */
.language-selector-label {{
    margin: 0 0 0.45rem 0;
    color: var(--text-main);
    font-size: 0.88rem;
    font-weight: 500;
}}
.language-toggle {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 0.55rem;
    width: 100%;
    margin: 0.15rem 0 0.9rem;
}}
.language-toggle a,
.language-toggle span.language-option {{
    min-width: 0;
    min-height: 2.7rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.42rem 0.56rem;
    border: 1px solid #d9dee8;
    border-radius: 12px;
    background: #fff;
    color: var(--text-main);
    text-decoration: none !important;
    font-size: 0.88rem;
    font-weight: 700;
    line-height: 1.05;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    overflow: hidden;
    transition: border-color 0.2s ease, background 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}}
.language-toggle a:hover {{
    border-color: var(--primary-red);
    background: #fff7f4;
    color: var(--text-main);
    box-shadow: 0 6px 16px rgba(255, 64, 17, 0.1);
    transform: translateY(-1px);
}}
.language-toggle .language-option.active {{
    border-color: var(--primary-red);
    background: #fff4ef;
    color: #c53114;
    box-shadow: inset 0 0 0 1px rgba(255, 64, 17, 0.22);
}}
.language-toggle .language-option.disabled {{
    cursor: not-allowed;
    opacity: 0.82;
}}
.language-toggle img {{
    width: 1.72rem;
    height: 1.15rem;
    flex: 0 0 auto;
    border-radius: 3px;
    object-fit: cover;
    box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.16);
}}
.language-toggle .language-text {{
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
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

/* Welcome Discover CTA */
.welcome-discover {{
    position: relative;
    margin: 28px auto 18px;
    padding: clamp(20px, 3vw, 32px) clamp(22px, 3.5vw, 40px);
    border-radius: 22px;
    background:
        radial-gradient(circle at 12% 18%, rgba(246, 218, 0, 0.32), transparent 55%),
        radial-gradient(circle at 92% 82%, rgba(255, 64, 17, 0.28), transparent 60%),
        linear-gradient(135deg, #fff7d6 0%, #ffe6df 100%);
    border: 1px solid rgba(255, 64, 17, 0.18);
    box-shadow: 0 18px 38px -22px rgba(255, 64, 17, 0.45);
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: clamp(16px, 3vw, 32px);
    overflow: hidden;
}}
.welcome-discover::before {{
    content: "";
    position: absolute;
    top: -40px;
    right: -40px;
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.55), transparent 70%);
    pointer-events: none;
}}
.welcome-discover-eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #c4321c;
    background: rgba(255, 255, 255, 0.65);
    padding: 4px 10px;
    border-radius: 999px;
    margin-bottom: 10px;
}}
.welcome-discover h3 {{
    margin: 0 0 6px;
    font-size: clamp(1.25rem, 2.4vw, 1.65rem);
    font-weight: 800;
    color: #1f2937;
    line-height: 1.25;
}}
.welcome-discover p {{
    margin: 0;
    font-size: 0.97rem;
    color: #3a4252;
    line-height: 1.55;
    max-width: 56ch;
}}
.welcome-discover-cta {{
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 14px 24px;
    border-radius: 14px;
    background: linear-gradient(135deg, #ff4011 0%, #ff7e3d 100%);
    color: #fff !important;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.01em;
    text-decoration: none !important;
    box-shadow: 0 12px 26px -10px rgba(255, 64, 17, 0.6);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    white-space: nowrap;
}}
.welcome-discover-cta:hover {{
    transform: translateY(-2px);
    box-shadow: 0 18px 32px -12px rgba(255, 64, 17, 0.7);
    color: #fff !important;
}}
.welcome-discover-cta svg {{
    width: 18px;
    height: 18px;
}}
@media (max-width: 760px) {{
    .welcome-discover {{
        grid-template-columns: 1fr;
        text-align: left;
    }}
    .welcome-discover-cta {{
        justify-self: start;
    }}
}}

@media (max-width: 420px) {{
    .language-toggle {{
        gap: 0.45rem;
    }}
    .language-toggle a,
    .language-toggle span.language-option {{
        min-height: 2.55rem;
        padding: 0.4rem 0.42rem;
        font-size: 0.82rem;
    }}
    .language-toggle img {{
        width: 1.45rem;
        height: 0.97rem;
    }}
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
    initial_sidebar_state="auto",
)


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
        "request_running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def sync_page_from_query_params() -> None:
    """Apply supported query parameters to the current UI state."""
    try:
        raw_page_value = st.query_params.get("page", "")
        raw_language_value = st.query_params.get("lang", "")
    except Exception as exc:
        logging.getLogger(__name__).debug("Could not read Streamlit query parameters: %s", exc)
        raw_page_value = ""
        raw_language_value = ""

    if isinstance(raw_page_value, list):
        raw_page_value = raw_page_value[0] if raw_page_value else ""
    if isinstance(raw_language_value, list):
        raw_language_value = raw_language_value[0] if raw_language_value else ""

    page_value = str(raw_page_value).strip().lower()
    language_value = str(raw_language_value).strip().lower()

    if page_value in {"chat", "info"}:
        st.session_state.current_page = page_value
    if language_value in LANGUAGE_OPTIONS:
        st.session_state.language = language_value


def build_sidebar_url(page: str, language: str) -> str:
    """Build a same-tab sidebar URL preserving page and language."""
    safe_page = page if page in {"chat", "info"} else "chat"
    safe_language = language if language in LANGUAGE_OPTIONS else "pt"
    return f"?page={safe_page}&lang={safe_language}"


def set_navigation_query_params(page: str) -> None:
    """Navigate between app pages while preserving the selected UI language."""
    safe_page = page if page in {"chat", "info"} else "chat"
    language = st.session_state.get("language", "pt")
    safe_language = language if language in LANGUAGE_OPTIONS else "pt"
    st.session_state.current_page = safe_page
    st.session_state.language = safe_language
    st.query_params["page"] = safe_page
    st.query_params["lang"] = safe_language


def render_language_selector(request_locked: bool) -> None:
    """Render a responsive language selector with stable ellipsis behaviour."""
    current_language = st.session_state.get("language", "pt")
    current_page = st.session_state.get("current_page", "chat")

    items = []
    for language_key, option in LANGUAGE_OPTIONS.items():
        is_active = language_key == current_language
        css_classes = ["language-option"]
        if is_active:
            css_classes.append("active")
        if request_locked:
            css_classes.append("disabled")

        label = html.escape(option["label"])
        short_label = html.escape(option["short"])
        image_uri = html.escape(FLAG_IMAGE_URIS[language_key], quote=True)
        content = (
            f'<img src="{image_uri}" alt="{short_label} flag" loading="lazy" />'
            f'<span class="language-text">{label}</span>'
        )

        class_attr = " ".join(css_classes)
        if request_locked:
            items.append(f'<span class="{class_attr}" aria-disabled="true">{content}</span>')
            continue

        href = html.escape(build_sidebar_url(current_page, language_key), quote=True)
        aria_current = ' aria-current="true"' if is_active else ""
        items.append(
            f'<a class="{class_attr}" href="{href}" target="_self"{aria_current}>{content}</a>'
        )

    selector_label = html.escape(t("language"))
    st.markdown(
        f"""
        <p class="language-selector-label">{selector_label}</p>
        <div class="language-toggle" role="group" aria-label="{selector_label}">
            {''.join(items)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _localized_kb_status(kb_ok: bool, language: str) -> str:
    """Return a localized knowledge-base readiness message."""
    if kb_ok:
        return "Base de conhecimento pronta." if language == "pt" else "Knowledge base ready."
    return (
        "Não foi possível carregar a base de conhecimento."
        if language == "pt"
        else "Could not load the knowledge base."
    )


def _localize_startup_status(status: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Copy shared startup status and localize session-facing messages."""
    localized_status = dict(status)
    localized_status["kb_status"] = _localized_kb_status(
        bool(localized_status.get("kb_ok", False)),
        language,
    )
    return localized_status


@st.cache_resource(
    show_spinner=False,
    ttl=Config.STREAMLIT_RESOURCE_CACHE_TTL_SECONDS,
)
def _load_shared_startup_resources() -> Dict[str, Any]:
    """Load shared runtime resources once per Streamlit server process.

    This intentionally caches only process-wide resources such as the vector
    store embeddings and generated transport databases. The conversational
    assistant remains in ``st.session_state`` because it is user/session scoped.
    """
    return _run_startup_preload_impl(language="en")


def _run_startup_preload(language: str = "pt") -> Dict[str, Any]:
    """Return one-time shared resource status localized for the active session."""
    return _localize_startup_status(_load_shared_startup_resources(), language)


def ensure_startup_resources(
    show_spinner: bool = True,
    force_retry: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Ensure one-time shared resources are loaded during app startup."""
    attempted = bool(st.session_state.get("startup_resources_attempted", False))
    cached_ok = st.session_state.get("startup_resources_ok")
    cached_status = st.session_state.get("startup_resources_status") or {}
    language = st.session_state.get("language", "pt")

    if attempted and cached_ok is not None and not force_retry:
        return bool(cached_ok), _localize_startup_status(cached_status, language)

    spinner_text = (
        "🚀 A preparar conhecimento e dados de mobilidade..."
        if language == "pt"
        else "🚀 Preparing knowledge base and mobility data..."
    )

    def _load() -> Tuple[bool, Dict[str, Any]]:
        if force_retry:
            _load_shared_startup_resources.clear()
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
    return re.sub(r"(Bearer\s+)[^\s'\"]+", r"\1[REDACTED]", sanitized)


def test_assistant_connection(provider: str) -> Tuple[bool, Optional[str]]:
    """Run a minimal inference request to confirm the selected model is ready."""
    lang = st.session_state.get("language", "pt")
    placeholder = st.empty()
    from agent.utils.model_connection_probe import perform_raw_model_connection_probe

    test_llm = st.session_state.assistant.supervisor.llm
    model_display = st.session_state.assistant.model_name

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
        from agent.graph import MultiAgentAssistant

        set_credentials_env(provider)

        startup_ok, startup_status = ensure_startup_resources(
            show_spinner=False,
            force_retry=bool(st.session_state.get("startup_resources_attempted"))
            and not bool(st.session_state.get("startup_resources_ok")),
        )
        transport_status = str(
            startup_status.get("transport_status")
            or st.session_state.get("transport_db_status")
            or ""
        )
        st.session_state.transport_db_status = transport_status

        if not startup_gate_allows_requests(
            startup_ok,
            startup_status,
        ):
            st.session_state.initialized = False
            return (
                False,
                build_startup_gate_message(
                    startup_status,
                    language=lang,
                ),
            )

        with st.spinner(
            "🤖 A iniciar o assistente..."
            if lang == "pt"
            else "🤖 Initializing assistant..."
        ):
            st.session_state.assistant = MultiAgentAssistant()

        if run_connection_probe:
            connection_ok, connection_error = test_assistant_connection(provider)
            if not connection_ok:
                st.session_state.assistant = None
                st.session_state.initialized = False
                return False, connection_error

        st.session_state.initialized = True
        st.session_state.provider = provider
        st.session_state.error = None

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

if small_logo_path:
    st.logo(small_logo_path, icon_image=logo_path or small_logo_path, size="small")


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
        request_locked = request_capture_locked(
            st.session_state.get("pending_request"),
            st.session_state.get("request_running", False),
        )

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
            disabled=request_locked,
        ):
            set_navigation_query_params("chat")
            st.rerun()
        if col2.button(
            "ℹ️ Info",
            use_container_width=True,
            type="primary" if st.session_state.current_page == "info" else "secondary",
            disabled=request_locked,
        ):
            set_navigation_query_params("info")
            st.rerun()

        st.divider()

        render_language_selector(request_locked=request_locked)

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
                    disabled=request_locked,
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
            if st.button(
                "🗑️ " + t("clear_conversation"),
                use_container_width=True,
                disabled=request_locked,
            ):
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
    info_href = html.escape(
        build_sidebar_url("info", st.session_state.get("language", "pt")),
        quote=True,
    )
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
        <div class="welcome-discover">
            <div>
                <span class="welcome-discover-eyebrow">✨ {t('discover_eyebrow')}</span>
                <h3>{t('discover_title')}</h3>
                <p>{t('discover_subtitle')}</p>
            </div>
            <a class="welcome-discover-cta" href="{info_href}" target="_self" rel="noopener">
                <span>ℹ️ {t('discover_cta')}</span>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <line x1="5" y1="12" x2="19" y2="12"/>
                    <polyline points="13 6 19 12 13 18"/>
                </svg>
            </a>
        </div>
        """,
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

    text = normalize_streamlit_chat_markdown(text)
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


def normalize_streamlit_chat_markdown(text: str) -> str:
    """Convert LISBOA card indentation into Markdown Streamlit renders as lists.

    Streamlit/CommonMark treats a standalone four-space-indented ``-`` line as
    a code block unless it is nested below a parent list item. LISBOA answers
    use indented field bullets for visual alignment, so the app adds a list
    parent for bold item cards and flattens any remaining orphan indented
    bullets before calling ``st.markdown``.

    Args:
        text: Final assistant Markdown.

    Returns:
        Markdown that preserves content while avoiding accidental code blocks.
    """
    if not text:
        return text or ""

    lines = text.splitlines()
    output: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if (
            re.match(r"^\*\*(?:🏷️|📍|🏛️|🍽️|☕|🥐|🎭)\s+[^*]+\*\*$", stripped)
            and re.match(r"^\s{4,}[-*]\s+", next_line)
        ):
            output.append(f"- {stripped}")
            continue
        if re.match(r"^\s{4,}[-*]\s+", line):
            previous = next((candidate.strip() for candidate in reversed(output) if candidate.strip()), "")
            if not previous.startswith(("- ", "* ")):
                output.append(re.sub(r"^\s{4,}([-*]\s+)", r"\1", line))
                continue
        output.append(line)
    normalized = "\n".join(output)
    if text.endswith("\n"):
        normalized += "\n"
    return normalized


def clean_response_for_display(text: str) -> str:
    """Remove citation artefacts and apply the final render-safe Markdown guard."""
    from agent.utils.response_formatter import (
        final_post_qa_guard,
        final_visual_pass,
        infer_response_language,
    )

    cleaned = re.sub(r"【.*?】", "", text or "")
    cleaned = cleaned.replace("\x00", "").strip()
    display_language = infer_response_language(
        context_text=cleaned,
        default=st.session_state.get("language", "en"),
    )
    return final_post_qa_guard(final_visual_pass(cleaned), language=display_language)


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
) -> bool:
    """Return whether the startup readiness gate is open for new user requests."""
    if not startup_ok or not bool(startup_status.get("ok", False)):
        return False
    if not bool(startup_status.get("transport_ok", False)):
        return False
    if not bool(startup_status.get("kb_ok", False)):
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


def attempt_startup_auto_initialization(selected_provider: str) -> bool:
    """Initialize the assistant automatically when production credentials are ready.

    Returns:
        True when initialization was attempted and the app requested a rerun.
        False when no startup initialization is needed.
    """
    credentials_ready, _ = provider_has_required_credentials(selected_provider)
    should_initialize = should_attempt_startup_auto_initialization(
        initialized=bool(st.session_state.get("initialized", False)),
        current_provider=str(st.session_state.get("provider") or ""),
        selected_provider=selected_provider,
        credentials_ready=credentials_ready,
        attempted_provider=st.session_state.get("startup_auto_init_attempted_provider"),
        last_error=st.session_state.get("startup_auto_init_error"),
    )
    if not should_initialize:
        return False

    st.session_state.request_running = True
    spinner_text = (
        "🚀 A preparar o LISBOA para a primeira pergunta..."
        if st.session_state.get("language", "pt") == "pt"
        else "🚀 Preparing LISBOA for the first prompt..."
    )
    try:
        with st.spinner(spinner_text):
            success, error = initialize_assistant(
                selected_provider,
                run_connection_probe=False,
            )
    finally:
        st.session_state.request_running = False

    st.session_state.startup_auto_init_attempted_provider = selected_provider
    if success:
        st.session_state.startup_auto_init_error = None
    else:
        st.session_state.startup_auto_init_error = error or t("initialization_failed")

    st.rerun()
    return True


def build_startup_gate_message(
    startup_status: Dict[str, Any],
    *,
    language: str,
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
    if not bool(startup_status.get("kb_ok", False)) and kb_status:
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


def request_capture_locked(
    pending_request: Optional[str],
    request_running: bool = False,
) -> bool:
    """Return whether the UI should temporarily block new requests."""
    return bool(pending_request) or bool(request_running)


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
    st.session_state.request_running = True
    st.session_state.messages.append({"role": "user", "content": user_input})


def run_info_page() -> None:
    """Render the visual project information page."""
    render_html_block("""
        <style>
        .info-main-container {
            max-width: 1240px;
            margin: 0 auto;
            padding: 0.4rem 0 0.5rem 0;
            animation: fadeIn 0.7s ease;
            container-type: inline-size;
        }

        .info-hero {
            position: relative;
            overflow: hidden;
            display: block;
            min-height: clamp(420px, 54vh, 585px);
            padding: clamp(1.4rem, 4vw, 3.7rem);
            margin-bottom: 1.2rem;
            border-radius: 8px;
            background:
                linear-gradient(90deg, rgba(13, 18, 32, 0.88) 0%, rgba(13, 18, 32, 0.62) 45%, rgba(13, 18, 32, 0.22) 100%),
                linear-gradient(180deg, rgba(13, 18, 32, 0.12) 0%, rgba(13, 18, 32, 0.75) 100%),
                var(--info-hero-image, linear-gradient(135deg, #111827, #334155));
            background-position: center;
            background-size: cover;
            box-shadow: 0 22px 60px rgba(15, 23, 42, 0.18);
            isolation: isolate;
        }

        .info-hero-copy {
            position: relative;
            z-index: 1;
        }

        .info-hero-copy {
            max-width: min(860px, 100%);
            padding-top: clamp(0.25rem, 3vh, 2.25rem);
        }

        .info-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            width: fit-content;
            padding: 0.48rem 0.72rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.36);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.14);
            color: #ffffff;
            font-size: 0.82rem;
            font-weight: 800;
            backdrop-filter: blur(10px);
        }

        .info-hero-copy h2 {
            max-width: 850px;
            margin: 0 0 0.65rem 0;
            color: #ffffff;
            font-size: clamp(2.75rem, 6.4vw, 6.5rem);
            line-height: 1;
            letter-spacing: 0;
            font-weight: 800;
            text-shadow: 0 18px 46px rgba(0, 0, 0, 0.34);
        }

        .info-hero-copy h3 {
            max-width: 760px;
            margin: 0 0 1.05rem 0;
            color: #fff3b0;
            font-size: clamp(1.12rem, 2.25vw, 1.55rem);
            font-weight: 800;
            letter-spacing: 0;
        }

        .info-hero-copy p {
            max-width: 760px;
            margin: 0;
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.08rem;
            line-height: 1.75;
        }

        .info-stat-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            max-width: 780px;
            margin-top: 1.45rem;
        }

        .info-stat {
            min-height: 96px;
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.22);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.13);
            backdrop-filter: blur(12px);
        }

        .info-stat:nth-child(2) {
            background: rgba(246, 218, 0, 0.2);
        }

        .info-stat:nth-child(3) {
            background: rgba(255, 64, 17, 0.18);
        }

        .info-stat-value {
            display: block;
            color: #ffffff;
            font-size: 1.75rem;
            line-height: 1;
            font-weight: 800;
        }

        .info-stat-label {
            display: block;
            margin-top: 0.42rem;
            color: rgba(255, 255, 255, 0.78);
            font-size: 0.88rem;
            font-weight: 700;
        }

        .info-layer-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: -2.1rem auto 1.4rem auto;
            position: relative;
            z-index: 2;
        }

        .info-card {
            min-height: 205px;
            padding: 1.15rem;
            border: 1px solid rgba(255, 255, 255, 0.68);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.94);
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.1);
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        }

        .info-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 22px 52px rgba(15, 23, 42, 0.1);
        }

        .info-card-red { border-top: 4px solid var(--primary-red); }
        .info-card-blue { border-top: 4px solid #3777ff; }
        .info-card-green { border-top: 4px solid #0ee071; }
        .info-card-yellow { border-top: 4px solid var(--primary-yellow); }

        .info-card-red:hover { border-color: rgba(255, 64, 17, 0.3); }
        .info-card-blue:hover { border-color: rgba(55, 119, 255, 0.3); }
        .info-card-green:hover { border-color: rgba(14, 224, 113, 0.32); }
        .info-card-yellow:hover { border-color: rgba(246, 218, 0, 0.42); }

        .info-card-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2.9rem;
            height: 2.9rem;
            margin-bottom: 1rem;
            border-radius: 8px;
            background: #f8fafc;
            font-size: 1.6rem;
        }

        .info-card-title {
            margin-bottom: 0.55rem;
            color: #111827;
            font-weight: 800;
            font-size: 1.05rem;
        }

        .info-card-desc {
            color: #5f6f82;
            font-size: 0.94rem;
            line-height: 1.65;
        }

        .info-system-band,
        .info-framework-section,
        .info-audience-section {
            margin: 1.5rem 0;
            padding: clamp(1.35rem, 2.4vw, 2.15rem);
            border-radius: 8px;
        }

        .info-section-heading {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-end;
            margin-bottom: 1.15rem;
        }

        .info-section-heading h3,
        .info-system-title {
            margin: 0;
            color: #111827;
            font-size: clamp(1.25rem, 2.2vw, 1.75rem);
            font-weight: 800;
            letter-spacing: 0;
        }

        .info-section-heading p,
        .info-system-desc {
            max-width: 980px;
            margin: 0.45rem 0 0 0;
            color: #526273;
            font-size: 0.98rem;
            line-height: 1.65;
        }

        .info-framework-section .info-section-heading p,
        .info-audience-section .info-section-heading p {
            max-width: min(1120px, 100%);
        }

        .info-system-band {
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            gap: clamp(1rem, 2.4vw, 1.6rem);
            align-items: stretch;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background:
                linear-gradient(135deg, rgba(255, 64, 17, 0.08), rgba(55, 119, 255, 0.08)),
                #ffffff;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.07);
        }

        .info-system-copy {
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 0.35rem;
            max-width: 880px;
        }

        .info-system-badge {
            display: inline-flex;
            width: fit-content;
            margin-bottom: 0.9rem;
            padding: 0.42rem 0.68rem;
            border-radius: 999px;
            color: #9a2a0c;
            background: rgba(255, 64, 17, 0.1);
            font-size: 0.78rem;
            font-weight: 800;
        }

        .info-flow-track {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .info-flow-step {
            min-height: 190px;
            padding: 1rem;
            border-radius: 8px;
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
        }

        .info-flow-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2.35rem;
            height: 2.35rem;
            margin-bottom: 0.85rem;
            border-radius: 8px;
            color: #ffffff;
            background: #111827;
            font-weight: 800;
        }

        .info-flow-red .info-flow-number { background: #c8320e; }
        .info-flow-yellow .info-flow-number { background: #705900; }
        .info-flow-blue .info-flow-number { background: #1f5fcf; }
        .info-flow-green .info-flow-number { background: #056f4b; }

        .info-flow-step h3 {
            margin: 0 0 0.45rem 0;
            color: #111827;
            font-size: 1rem;
            font-weight: 800;
        }

        .info-flow-step p {
            margin: 0;
            color: #5f6f82;
            font-size: 0.9rem;
            line-height: 1.6;
        }

        .info-framework-viewport {
            overflow: hidden;
            padding: 0.85rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.07);
        }

        .info-framework-image {
            display: block;
            width: 100%;
            min-width: 0;
            height: auto;
            max-height: none;
            object-fit: contain;
            border-radius: 8px;
        }

        .info-source-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
        }

        .info-source-link {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            min-height: 76px;
            padding: 0.8rem 0.9rem;
            border: 1px solid rgba(55, 119, 255, 0.14);
            border-radius: 8px;
            background: linear-gradient(135deg, rgba(55, 119, 255, 0.06), rgba(14, 224, 113, 0.05));
            color: #111827;
            text-decoration: none;
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .info-source-link:hover {
            transform: translateY(-2px);
            border-color: rgba(55, 119, 255, 0.32);
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.08);
            color: #111827;
        }

        .info-source-link strong {
            display: block;
            color: #111827;
            font-size: 0.96rem;
            line-height: 1.2;
        }

        .info-source-link small {
            display: block;
            margin-top: 0.18rem;
            color: #5f6f82;
            font-size: 0.78rem;
            line-height: 1.35;
        }

        .info-example-section {
            margin: 1.5rem 0;
            padding: clamp(1.35rem, 2.4vw, 2.15rem);
            border-radius: 8px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(248, 250, 252, 0.95)),
                linear-gradient(90deg, rgba(255, 64, 17, 0.06), rgba(55, 119, 255, 0.06));
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.07);
        }

        .info-example-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .info-example-card {
            min-height: 180px;
            padding: 1rem;
            border-radius: 8px;
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.055);
        }

        .info-example-red { border-top: 4px solid var(--primary-red); }
        .info-example-blue { border-top: 4px solid #3777ff; }
        .info-example-yellow { border-top: 4px solid var(--primary-yellow); }
        .info-example-green { border-top: 4px solid #0ee071; }

        .info-example-title {
            display: flex;
            gap: 0.5rem;
            align-items: center;
            margin-bottom: 0.8rem;
            color: #111827;
            font-size: 0.98rem;
            font-weight: 800;
            line-height: 1.35;
        }

        .info-example-card ul {
            margin: 0;
            padding: 0;
            list-style: none;
        }

        .info-example-card li {
            position: relative;
            margin: 0 0 0.68rem 0;
            padding-left: 1rem;
            color: #526273;
            font-size: 0.88rem;
            line-height: 1.48;
            overflow-wrap: anywhere;
        }

        .info-example-card li:last-child {
            margin-bottom: 0;
        }

        .info-example-card li::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0.58em;
            width: 0.38rem;
            height: 0.38rem;
            border-radius: 999px;
            background: var(--primary-red);
        }

        .info-audience-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
        }

        .info-audience-card {
            min-height: 160px;
            padding: 1.3rem;
            border-radius: 8px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(248, 250, 252, 0.9)),
                radial-gradient(circle at top right, rgba(246, 218, 0, 0.2), transparent 38%);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
        }

        .info-audience-card strong {
            display: block;
            margin-bottom: 0.45rem;
            color: #111827;
            font-size: 1.08rem;
        }

        .info-audience-card p {
            margin: 0;
            color: #5f6f82;
            line-height: 1.65;
        }

        .info-audience-card p::before {
            content: "";
            display: block;
            width: 2.6rem;
            height: 3px;
            margin-bottom: 0.8rem;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--primary-red), var(--primary-yellow));
        }

        .info-details-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1rem;
            margin: 1.5rem 0;
        }

        .info-detail-card {
            padding: 1.35rem;
            border-radius: 8px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: #ffffff;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
        }

        .info-detail-wide {
            grid-column: 1 / -1;
        }

        .info-detail-red { border-left: 4px solid var(--primary-red); }
        .info-detail-blue { border-left: 4px solid #3777ff; }
        .info-detail-green { border-left: 4px solid #0ee071; }
        .info-detail-yellow { border-left: 4px solid var(--primary-yellow); }

        .info-detail-title {
            display: flex;
            gap: 0.6rem;
            align-items: center;
            margin-bottom: 0.9rem;
            color: #111827;
            font-size: 1.05rem;
            font-weight: 800;
        }

        .info-detail-body {
            color: #334155;
            font-size: 0.94rem;
            line-height: 1.68;
        }

        .info-detail-body p { margin: 0 0 0.75rem 0; }
        .info-detail-body p:last-child { margin-bottom: 0; }
        .info-detail-body ul,
        .info-detail-body ol { margin: 0; padding-left: 1.2rem; }
        .info-detail-body li { margin-bottom: 0.42rem; color: #5f6f82; }
        .info-detail-body strong { color: #111827; }

        .info-footer {
            margin-top: 1.75rem;
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 1.2rem;
            align-items: center;
            padding: clamp(1.2rem, 2.4vw, 1.7rem);
            border: 1px solid rgba(177, 177, 177, 0.18);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(246, 218, 0, 0.08), rgba(255, 64, 17, 0.08), rgba(55, 119, 255, 0.08) 52%, rgba(14, 224, 113, 0.07)),
                #ffffff;
            box-shadow: 0 18px 42px rgba(22, 29, 45, 0.075);
            text-align: left;
        }

        .info-author-label {
            display: inline-flex;
            margin-bottom: 0.35rem;
            color: #a8320f;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
        }

        .info-author-name {
            margin: 0;
            color: #1f2937;
            font-size: clamp(1.35rem, 2.4vw, 1.75rem);
            line-height: 1.15;
            font-weight: 800;
            letter-spacing: 0;
        }

        .info-author-project {
            margin: 0.45rem 0 0 0;
            color: #526273;
            font-size: 0.98rem;
            line-height: 1.45;
        }

        .info-author-meta {
            margin: 0.65rem 0 0 0;
            color: #334155;
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .info-author-secondary {
            display: block;
            margin-top: 0.08rem;
            color: #64748b;
            font-size: 0.8rem;
            line-height: 1.45;
        }

        .info-author-links {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 0.6rem;
        }

        .info-author-link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.45rem;
            min-height: 44px;
            padding: 0.62rem 0.86rem;
            border-radius: 8px;
            border: 1px solid rgba(255, 64, 17, 0.18);
            background: rgba(255, 255, 255, 0.86);
            color: #111827;
            font-size: 0.9rem;
            font-weight: 800;
            text-decoration: none;
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .info-author-link:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 64, 17, 0.24);
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
            color: #111827;
        }

        .info-author-link::before {
            content: "";
            width: 1rem;
            height: 1rem;
            flex: 0 0 auto;
            background-repeat: no-repeat;
            background-position: center;
            background-size: contain;
        }

        .info-author-link-github::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23111827' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5 .08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.4 5.4 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4'/%3E%3Cpath d='M9 18c-4.51 2-5-2-7-2'/%3E%3C/svg%3E");
        }

        .info-author-link-linkedin::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23111827' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-4 0v7h-4v-7a6 6 0 0 1 6-6z'/%3E%3Crect width='4' height='12' x='2' y='9'/%3E%3Ccircle cx='4' cy='4' r='2'/%3E%3C/svg%3E");
        }

        .back-btn-container { margin-top: 2rem; display: flex; justify-content: center; }

        @media (max-width: 980px) {
            .info-layer-grid,
            .info-system-band,
            .info-flow-track,
            .info-example-grid {
                grid-template-columns: 1fr 1fr;
            }
            .info-source-grid { grid-template-columns: 1fr 1fr; }
            .info-details-grid { grid-template-columns: 1fr; }
            .info-system-band { grid-template-columns: 1fr; }
            .info-footer {
                grid-template-columns: 1fr;
            }
            .info-author-links {
                justify-content: flex-start;
            }
        }

        @media (max-width: 720px) {
            .info-main-container { padding-top: 0.25rem; }
            .info-layer-grid,
            .info-system-band,
            .info-flow-track,
            .info-audience-grid,
            .info-details-grid,
            .info-example-grid,
            .info-stat-grid {
                grid-template-columns: 1fr;
            }
            .info-source-grid { grid-template-columns: 1fr; }
            .info-hero { min-height: 530px; }
            .info-hero-copy { max-width: 100%; padding-top: 0; }
            .info-layer-grid { margin-top: 0.8rem; }
            .info-footer { grid-template-columns: 1fr; }
        }

        @container (max-width: 900px) {
            .info-hero {
                min-height: auto;
                padding: 1.65rem;
            }

            .info-hero-copy {
                max-width: 100%;
                padding-top: 0;
            }

            .info-hero-copy h2 {
                font-size: clamp(3.25rem, 10cqw, 4.9rem);
            }

            .info-layer-grid,
            .info-flow-track {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .info-layer-grid {
                margin-top: 0.95rem;
            }

            .info-flow-step,
            .info-card {
                min-height: auto;
            }

            .info-example-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @container (max-width: 700px) {
            .info-hero {
                min-height: 530px;
                padding: 1.35rem;
            }

            .info-hero-copy {
                max-width: 100%;
                padding-top: 0;
            }

            .info-hero-copy h2 {
                font-size: clamp(2.35rem, 12cqw, 3.15rem);
            }

            .info-hero-copy h3 {
                font-size: 1.08rem;
            }

            .info-hero-copy p {
                font-size: 0.98rem;
            }

            .info-layer-grid,
            .info-system-band,
            .info-flow-track,
            .info-audience-grid,
            .info-details-grid,
            .info-source-grid,
            .info-example-grid,
            .info-stat-grid {
                grid-template-columns: 1fr;
            }

            .info-layer-grid {
                margin-top: 0.85rem;
            }

            .info-card,
            .info-flow-step,
            .info-example-card {
                min-height: auto;
            }

            .info-details-grid {
                gap: 0.85rem;
            }
        }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        </style>
    """)

    hero_style = (
        f' style="--info-hero-image: url(&quot;{html.escape(banner_url, quote=True)}&quot;);"'
        if banner_url
        else ""
    )

    stat_items = [
        (t("info_stat_agents_value"), t("info_stat_agents_label")),
        (t("info_stat_tools_value"), t("info_stat_tools_label")),
        (t("info_stat_scope_value"), t("info_stat_scope_label")),
    ]
    stat_cards = "".join(
        "<div class=\"info-stat\">"
        f"<span class=\"info-stat-value\">{html.escape(value)}</span>"
        f"<span class=\"info-stat-label\">{html.escape(label)}</span>"
        "</div>"
        for value, label in stat_items
    )

    feature_cards = "".join(
        [
            build_info_feature_card_html("🏛️", t("info_f1_title"), t("info_f1_desc"), "red"),
            build_info_feature_card_html("🚇", t("info_f2_title"), t("info_f2_desc"), "blue"),
            build_info_feature_card_html("🌤️", t("info_f3_title"), t("info_f3_desc"), "yellow"),
            build_info_feature_card_html("🏥", t("info_f4_title"), t("info_f4_desc"), "green"),
        ]
    )

    flow_items = [
        ("01", t("info_flow_1_title"), t("info_flow_1_desc"), "red"),
        ("02", t("info_flow_2_title"), t("info_flow_2_desc"), "yellow"),
        ("03", t("info_flow_3_title"), t("info_flow_3_desc"), "blue"),
        ("04", t("info_flow_4_title"), t("info_flow_4_desc"), "green"),
    ]
    flow_cards = "".join(
        f'<article class="info-flow-step info-flow-{tone}">'
        f'<span class="info-flow-number">{number}</span>'
        f'<h3>{html.escape(title)}</h3>'
        f'<p>{html.escape(description)}</p>'
        '</article>'
        for number, title, description, tone in flow_items
    )

    source_items = [
        ("VisitLisboa", t("info_source_visitlisboa_desc"), "https://www.visitlisboa.com/"),
        ("IPMA", t("info_source_ipma_desc"), "https://ipma.pt/"),
        ("Metro", t("info_source_metro_desc"), "https://www.metrolisboa.pt/"),
        ("Carris", t("info_source_carris_desc"), "https://www.carris.pt/"),
        ("Carris Metropolitana", t("info_source_cm_desc"), "https://www.carrismetropolitana.pt/"),
        ("CP", t("info_source_cp_desc"), "https://www.cp.pt/"),
        ("Lisboa Aberta", t("info_source_lisboa_aberta_desc"), "https://dados.cm-lisboa.pt/"),
    ]
    source_links = (
        '<div class="info-source-grid">'
        + "".join(build_info_source_link_html(label, description, url) for label, description, url in source_items)
        + "</div>"
    )

    example_items = [
        ("🌦️", t("info_example_weather_title"), t_list("info_example_weather_items"), "yellow"),
        ("🚇", t("info_example_mobility_title"), t_list("info_example_mobility_items"), "blue"),
        ("🏛️", t("info_example_tourism_title"), t_list("info_example_tourism_items"), "red"),
        ("🗺️", t("info_example_plan_title"), t_list("info_example_plan_items"), "green"),
    ]
    example_cards = "".join(
        build_info_example_card_html(icon, title, examples, tone)
        for icon, title, examples, tone in example_items
    )
    examples_markup = (
        '<section class="info-example-section">'
        '<div class="info-section-heading">'
        '<div>'
        f'<h3>💬 {html.escape(t("info_examples_title"))}</h3>'
        f'<p>{html.escape(t("info_examples_desc"))}</p>'
        '</div>'
        '</div>'
        f'<div class="info-example-grid">{example_cards}</div>'
        '</section>'
    )

    detail_cards = (
        build_info_detail_card_html(
            "🧩",
            t("info_data_sources"),
            source_links,
            "blue",
            "info-detail-wide",
        )
        + build_info_detail_card_html(
            "🧭",
            t("info_scope_title"),
            build_info_list_html(t_list("info_scope_items")),
            "yellow",
            "info-detail-wide",
        )
    )

    framework_markup = ""
    if framework_url:
        framework_markup = (
            '<section class="info-framework-section">'
            '<div class="info-section-heading">'
            '<div>'
            f'<h3>🧭 {html.escape(t("info_framework_title"))}</h3>'
            f'<p>{html.escape(t("info_framework_desc"))}</p>'
            '</div>'
            '</div>'
            '<div class="info-framework-viewport">'
            f'<img class="info-framework-image" src="{framework_url}" '
            'alt="LISBOA multi-agent system architecture diagram">'
            '</div>'
            '</section>'
        )

    audience_markup = (
        '<section class="info-audience-section">'
        '<div class="info-section-heading">'
        '<div>'
        f'<h3>👥 {html.escape(t("info_audience_title"))}</h3>'
        f'<p>{html.escape(t("info_audience_desc"))}</p>'
        '</div>'
        '</div>'
        '<div class="info-audience-grid">'
        '<article class="info-audience-card">'
        f'<strong>🧳 {html.escape(t("info_tourists_title"))}</strong>'
        f'<p>{html.escape(t("info_tourists_desc"))}</p>'
        '</article>'
        '<article class="info-audience-card">'
        f'<strong>🏠 {html.escape(t("info_residents_title"))}</strong>'
        f'<p>{html.escape(t("info_residents_desc"))}</p>'
        '</article>'
        '</div>'
        '</section>'
    )

    html_content = (
        '<div class="info-main-container">'
        f'<section class="info-hero"{hero_style}>'
        '<div class="info-hero-copy">'
        f'<span class="info-kicker">✦ {html.escape(t("info_badge"))}</span>'
        f'<h2>{html.escape(t("info_title"))}</h2>'
        f'<h3>{html.escape(t("info_subtitle"))}</h3>'
        f'<p>{html.escape(t("info_intro"))}</p>'
        f'<div class="info-stat-grid">{stat_cards}</div>'
        '</div>'
        '</section>'
        f'<div class="info-layer-grid">{feature_cards}</div>'
        f'{examples_markup}'
        '<section class="info-system-band">'
        '<div class="info-system-copy">'
        f'<span class="info-system-badge">⚙️ {html.escape(t("info_architecture_title"))}</span>'
        f'<h3 class="info-system-title">{html.escape(t("info_flow_title"))}</h3>'
        f'<p class="info-system-desc">{md_to_html(html.escape(t("info_architecture_desc")))}</p>'
        '</div>'
        f'<div class="info-flow-track">{flow_cards}</div>'
        '</section>'
        f'{framework_markup}'
        f'<div class="info-details-grid">{detail_cards}</div>'
        f'{audience_markup}'
        '<div class="info-footer">'
        '<div class="info-author-copy">'
        f'<span class="info-author-label">🎓 {html.escape(t("info_author"))}</span>'
        '<h3 class="info-author-name">André Filipe Gomes Silvestre</h3>'
        f'<p class="info-author-project">{html.escape(t("info_author_project"))}</p>'
        f'<p class="info-author-meta"><strong>{html.escape(t("info_author_role"))}</strong> · '
        f'{html.escape(t("info_author_degree"))}'
        f'<span class="info-author-secondary">{html.escape(t("info_author_affiliation"))}</span>'
        f'<span class="info-author-secondary">{html.escape(t("info_author_year"))}</span></p>'
        '</div>'
        '<div class="info-author-links">'
        f'<a class="info-author-link info-author-link-github" href="https://github.com/Silvestre17" target="_blank" rel="noopener noreferrer"><span>{html.escape(t("info_author_github"))}</span></a>'
        f'<a class="info-author-link info-author-link-linkedin" href="https://www.linkedin.com/in/andrefgsilvestre/" target="_blank" rel="noopener noreferrer"><span>{html.escape(t("info_author_linkedin"))}</span></a>'
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
            set_navigation_query_params("chat")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================================================
# MAIN EXECUTION
# ==========================================================================


def main():
    init_system_state()
    st.markdown(CSS, unsafe_allow_html=True)
    sync_page_from_query_params()

    selected_provider, q_act = build_sidebar()

    if st.session_state.current_page == "info":
        run_info_page()
        return

    display_banner()
    attempt_startup_auto_initialization(selected_provider)

    pending = st.session_state.get("pending_request")
    request_locked = request_capture_locked(
        pending,
        st.session_state.get("request_running", False),
    )

    # Stage 1: Capture a new request from quick-action, chat input, or welcome
    # button. Queue it and append the user turn immediately, then rerun once so
    # the sidebar counter and chat history both reflect the new turn before the
    # assistant call starts.
    welcome_request = None
    chat_request = None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if msg["role"] == "assistant":
                content = normalize_streamlit_chat_markdown(content)
            st.markdown(content)

    if not request_locked and not st.session_state.messages:
        welcome_request = build_welcome()

    if in_text := st.chat_input(t("chat_placeholder"), disabled=request_locked):
        chat_request = in_text

    new_request = select_new_request(
        sidebar_request=q_act or None,
        welcome_request=welcome_request,
        chat_request=chat_request,
        pending_request=pending,
        allow_requests=not request_locked,
    )

    if new_request:
        queue_pending_request(new_request)
        st.rerun()

    # Stage 2: If a request is pending, ensure the assistant is initialized and
    # execute the LLM call. The user message has already been appended during
    # the previous rerun, so the sidebar counter is already in sync.
    pending = st.session_state.get("pending_request")
    if pending:
        st.session_state.request_running = True
        if (
            not st.session_state.initialized
            or st.session_state.provider != selected_provider
        ):
            success, error = initialize_assistant(
                selected_provider,
                run_connection_probe=False,
            )
            if not success:
                st.session_state.request_running = False
                st.session_state.pop("pending_request", None)
                st.session_state.pop("pending_request_user_appended", None)
                st.error(error or t("initialization_failed"))
                return
        already_appended = bool(st.session_state.pop("pending_request_user_appended", False))
        st.session_state.pop("pending_request", None)
        try:
            run_interaction(pending, user_message_already_rendered=already_appended)
        finally:
            st.session_state.request_running = False
        # Trigger one final rerun so the sidebar counter picks up the assistant
        # turn immediately instead of waiting for the next user action.
        st.rerun()

    if not st.session_state.initialized:
        credentials_ready, _ = provider_has_required_credentials(selected_provider)
        auto_init_error = st.session_state.get("startup_auto_init_error")
        if runtime_auto_initialize_enabled() and auto_init_error:
            st.error(auto_init_error)
        elif credentials_ready:
            ready_message = (
                "As credenciais já estão prontas. Envie uma pergunta para iniciar o assistente."
                if st.session_state.language == "pt"
                else "Your credentials are ready. Send a prompt to start the assistant."
            )
            if not runtime_auto_initialize_enabled():
                ready_message = (
                    "As credenciais já estão prontas. Pode clicar em **Ligar Sistema** na barra lateral ou enviar uma pergunta para iniciar automaticamente."
                    if st.session_state.language == "pt"
                    else "Your credentials are ready. Click **Connect System** in the sidebar or send a prompt to start automatically."
                )
            st.info(ready_message)
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
