# ==========================================================================
# Master Thesis - Streamlit Application
#   - André Filipe Gomes Silvestre, 20240502
#
#   Streamlit user interface for LISBOA, the Lisbon Itinerary System Based On
#   AI. Provides the interactive chat workflow, per-session model settings,
#   quick actions, tracing controls, debug visibility, and response rendering
#   for the multi-agent runtime.
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
import traceback
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

LISBON_TIMEZONE_NAME = "Europe/Lisbon"
THESIS_URL = "https://hdl.handle.net/10362/207399"
ARTICLE_DOI_URL = "https://doi.org/10.1016/j.rineng.2026.113369"
SOFTWARE_DOI_URL = "https://doi.org/10.5281/zenodo.23322220"


def get_lisbon_timezone() -> ZoneInfo | timezone:
    """Return Portugal Continental timezone for UI timestamps."""
    try:
        return ZoneInfo(LISBON_TIMEZONE_NAME)
    except ZoneInfoNotFoundError:
        # Last-resort hosted fallback. The tzdata package in requirements should
        # normally make this branch unnecessary.
        return timezone(timedelta(hours=1), name="WEST")


def format_lisbon_now() -> str:
    """Format the current Portugal Continental time for the sidebar footer."""
    return datetime.now(get_lisbon_timezone()).strftime("%Y-%m-%d %H:%M")


sys.path.insert(0, ".")

from agent.llm_factory import LLMConnection, LLMFactory
from agent.utils.model_connection_probe import check_tool_calling
from agent.utils.startup_resources import run_startup_preload as _run_startup_preload_impl
from agent.utils.startup_resources import schedule_runtime_data_refresh
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
    "pt": {"label": "Português"},
    "en": {"label": "English"},
}

# ==========================================================================
# TRANSLATIONS / INTERNATIONALIZATION
# ==========================================================================


TRANSLATIONS = {
    "en": {
        "app_title": "LISBOA",
        "app_subtitle": "Lisbon Itinerary System Based On AI",
        "settings": "System Settings",
        "language": "Language",
        "llm_active_server": "Active model: LISBOA default.",
        "llm_active_custom": "Active model: {model} ({host}).",
        "llm_server_unavailable": "The LISBOA default model is not available right now. Add your own provider below to continue.",
        "llm_preset": "Service",
        "llm_preset_other": "Other",
        "llm_base_url": "Base URL",
        "llm_api_key": "API key",
        "llm_model": "Model",
        "llm_model_help": "Any OpenAI-compatible model that supports tool calling.",
        "llm_key_privacy": "🔐 Your key is used only in this browser session and is not stored.",
        "llm_save": "Save and test",
        "llm_use_default": "Use LISBOA default",
        "llm_checking": "Testing the model...",
        "llm_check_ok": "Connected to {model}. Your next messages use this model.",
        "llm_default_restored": "Back to the LISBOA default model.",
        "llm_error_invalid_base_url": "Enter a valid base URL that starts with https:// or http://.",
        "llm_error_missing_model": "Enter the model name.",
        "llm_error_private_host": "This address is not allowed on the hosted app. Use the provider's public URL.",
        "llm_error_auth": "The provider rejected the API key. Check the key and try again.",
        "llm_error_not_found": "Model or URL not found. Check the base URL and the model name.",
        "llm_error_rate_limit": "The provider refused the request because of a rate or quota limit. Try again later.",
        "llm_error_connection": "Could not reach the provider. Check the base URL.",
        "llm_error_bad_request": "The provider rejected the test request. Check the model name and that the model supports tool calling.",
        "llm_error_no_tool_call": "The model answered but did not use the test tool. LISBOA needs a model with tool calling.",
        "llm_error_unknown": "The model test failed. Check the settings and try again.",
        "llm_setup_prompt": "To start, add your own AI provider in ⚙️ System Settings in the sidebar.",
        "llm_server_setup_prompt": "Configure the production credentials in environment variables or Streamlit secrets to get started.",
        "llm_ready_prompt": "The model is ready. Send a prompt to start the assistant.",
        "llm_switch_hint": "You can also use your own AI provider in ⚙️ System Settings.",
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
        "ex_query_planning": "Plan a short architecture walk through Baixa and Chiado.",
        "error_generic": "Service temporarily unavailable. Please try again later.",
        "history_window_notice": "Showing the last {count} messages to keep the interface responsive. The full conversation is still kept for the assistant.",
        "thinking": "Analyzing live city data...",
        "footer_version": "LISBOA | AI Assistant",
        "footer_made": "André Filipe Gomes Silvestre • NOVA IMS",
        "info_title": "LISBOA",
        "info_subtitle": "Lisbon Itinerary System Based On AI",
        "info_intro": "An academic multi-agent prototype that helps users ask grounded questions about Lisbon tourism, mobility, weather, local services, and itineraries.",
        "info_badge": "Master's Thesis and Published Article",
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
        "info_author_role": "MSc Graduate",
        "info_author_degree": "Data Science and Advanced Analytics",
        "info_author_affiliation": "NOVA IMS - Universidade NOVA de Lisboa",
        "info_author_year": "Academic year 2025/2026",
        "info_supervisors_label": "Supervisors",
        "info_supervisors_names": "Prof. Dr. Bruno Jardim   •   Prof. Dr. Miguel de Castro Neto",
        "info_author_github": "GitHub",
        "info_author_linkedin": "LinkedIn",
        "info_author_thesis": "Master's Thesis",
        "info_author_article": "Article",
        "info_author_software": "Software DOI",
        "info_publication_label": "Published in",
        "discover_eyebrow": "New here?",
        "discover_title": "Discover how LISBOA works",
        "discover_subtitle": "A multi-agent system grounding live transport, weather, events and tourism data into feasible itineraries for Lisbon.",
        "discover_cta": "Explore the system",
    },
    "pt": {
        "app_title": "LISBOA",
        "app_subtitle": "Lisbon Itinerary System Based On AI",
        "settings": "Configurações",
        "language": "Idioma / Language",
        "llm_active_server": "Modelo ativo: predefinido do LISBOA.",
        "llm_active_custom": "Modelo ativo: {model} ({host}).",
        "llm_server_unavailable": "O modelo predefinido do LISBOA não está disponível de momento. Adicione o seu próprio fornecedor abaixo para continuar.",
        "llm_preset": "Serviço",
        "llm_preset_other": "Outro",
        "llm_base_url": "URL base",
        "llm_api_key": "Chave de API",
        "llm_model": "Modelo",
        "llm_model_help": "Qualquer modelo compatível com OpenAI que suporte chamadas de ferramentas (tool calling).",
        "llm_key_privacy": "🔐 A sua chave só é usada nesta sessão do browser e não fica guardada.",
        "llm_save": "Guardar e testar",
        "llm_use_default": "Usar o modelo predefinido",
        "llm_checking": "A testar o modelo...",
        "llm_check_ok": "Ligado a {model}. As próximas mensagens usam este modelo.",
        "llm_default_restored": "Voltou ao modelo predefinido do LISBOA.",
        "llm_error_invalid_base_url": "Indique um URL base válido, começado por https:// ou http://.",
        "llm_error_missing_model": "Indique o nome do modelo.",
        "llm_error_private_host": "Este endereço não é permitido na aplicação alojada. Use o URL público do fornecedor.",
        "llm_error_auth": "O fornecedor recusou a chave de API. Verifique a chave e tente novamente.",
        "llm_error_not_found": "Modelo ou URL não encontrado. Verifique o URL base e o nome do modelo.",
        "llm_error_rate_limit": "O fornecedor recusou o pedido por limite de pedidos ou de quota. Tente novamente mais tarde.",
        "llm_error_connection": "Não foi possível contactar o fornecedor. Verifique o URL base.",
        "llm_error_bad_request": "O fornecedor recusou o pedido de teste. Verifique o nome do modelo e se este suporta chamadas de ferramentas.",
        "llm_error_no_tool_call": "O modelo respondeu, mas não usou a ferramenta de teste. O LISBOA precisa de um modelo com chamadas de ferramentas.",
        "llm_error_unknown": "O teste do modelo falhou. Verifique os dados e tente novamente.",
        "llm_setup_prompt": "Para começar, adicione o seu próprio fornecedor de IA em ⚙️ Configurações, na barra lateral.",
        "llm_server_setup_prompt": "Configure as credenciais de produção nas variáveis de ambiente ou nos Streamlit secrets para começar.",
        "llm_ready_prompt": "O modelo está pronto. Envie uma pergunta para iniciar o assistente.",
        "llm_switch_hint": "Também pode usar o seu próprio fornecedor de IA em ⚙️ Configurações.",
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
        "ex_query_planning": "Planeia uma caminhada curta de arquitetura pela Baixa e Chiado.",
        "error_generic": "Serviço temporariamente indisponível. Tente novamente mais tarde.",
        "history_window_notice": "A mostrar apenas as últimas {count} mensagens para manter a interface fluida. A conversa completa continua disponível para o assistente.",
        "thinking": "A processar dados urbanos...",
        "footer_version": "LISBOA | Assistente IA",
        "footer_made": "André Filipe Gomes Silvestre • NOVA IMS",
        "info_title": "LISBOA",
        "info_subtitle": "Lisbon Itinerary System Based On AI",
        "info_intro": "Um protótipo académico multi-agente que ajuda a fazer perguntas fundamentadas sobre turismo, mobilidade, meteorologia, serviços locais e roteiros em Lisboa.",
        "info_badge": "Tese de Mestrado e Artigo Publicado",
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
        "info_author_role": "Mestre",
        "info_author_degree": "Data Science and Advanced Analytics",
        "info_author_affiliation": "NOVA IMS - Universidade NOVA de Lisboa",
        "info_author_year": "Ano letivo 2025/2026",
        "info_supervisors_label": "Orientadores",
        "info_supervisors_names": "Prof. Dr. Bruno Jardim      •      Prof. Dr. Miguel de Castro Neto",
        "info_author_github": "GitHub",
        "info_author_linkedin": "LinkedIn",
        "info_author_thesis": "Tese de Mestrado",
        "info_author_article": "Artigo",
        "info_author_software": "DOI do software",
        "info_publication_label": "Publicado em",
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
    """Convert minimal inline markdown to safe HTML for use in rendered blocks.

    The input is HTML-escaped here, so callers must pass raw (unescaped) text.
    """
    html_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html.escape(text))
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
    --font-ui: 'Inter', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif;
    --font-display: 'Montserrat', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif;
}}

/* Base Fonts */
html, body, [class*="css"]  {{
    font-family: var(--font-ui);
}}
h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
    font-family: var(--font-display) !important;
}}

[data-testid="stSpinner"],
[data-testid="stSpinner"] * {{
    font-family: var(--font-ui) !important;
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

/* Hide Streamlit's hover anchor-link icons on all headings for a cleaner UI.
   These inline elements also disrupted the banner title layout. */
[data-testid="stHeaderActionElements"] {{ display: none !important; }}

/* Custom Banner Component */
/* Wrapper establishes a container context so the banner scales to the
   available content width (not the viewport), which is what matters in
   Streamlit where the sidebar reduces the usable width. */
.top-banner {{
    container-type: inline-size;
    display: block;
    width: 100%;
}}
.top-banner-container {{
    background: linear-gradient(135deg, rgba(255, 64, 17, 0.35) 0%, rgba(246, 218, 0, 0.25) 30%,
    rgba(14, 224, 113, 0.20) 70%, rgba(55, 119, 255, 0.25) 100%),
    linear-gradient(90deg, rgba(19, 19, 19, 0.35) 0%, rgba(23, 22, 19, 0.35) 100%),
    url('{banner_url}');
    background-size: cover;
    background-position: center 36%;
    border-radius: clamp(14px, 2.5cqi, 20px);
    padding: 2.35rem 1.45rem;
    min-height: 11.5rem;
    margin-bottom: 2rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 18px 42px rgba(255, 64, 17, 0.23), 0 3px 12px rgba(0,0,0,0.12);
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
    padding: 0;
    font-size: 3rem;
    line-height: 0.94;
    font-weight: 800;
    text-shadow: 0 4px 14px rgba(0,0,0,0.26);
    letter-spacing: 0;
    overflow-wrap: break-word;
    position: relative;
    z-index: 1;
}}

.top-banner-container p {{
    color: rgba(255,255,255,0.95);
    font-size: 1.04rem;
    margin: 0.72rem 0 0 0;
    font-weight: 600;
    line-height: 1.28;
    max-width: 42rem;
    text-shadow: 0 2px 8px rgba(0,0,0,0.26);
    position: relative;
    z-index: 1;
}}

@container (min-width: 340px) {{
    .top-banner-container {{
        min-height: 12.75rem;
        padding: 2.7rem 1.85rem;
    }}
    .top-banner-container h1 {{
        font-size: 3.85rem;
    }}
    .top-banner-container p {{
        font-size: 1.12rem;
    }}
}}

@container (min-width: 560px) {{
    .top-banner-container {{
        min-height: 15.25rem;
        padding: 3.25rem 2.7rem;
    }}
    .top-banner-container h1 {{
        font-size: 5.3rem;
    }}
    .top-banner-container p {{
        font-size: 1.38rem;
    }}
}}

@container (min-width: 780px) {{
    .top-banner-container {{
        min-height: 17.25rem;
        padding: 3.9rem 3.5rem;
    }}
    .top-banner-container h1 {{
        font-size: 6.35rem;
    }}
    .top-banner-container p {{
        font-size: 1.55rem;
    }}
}}

/* Sidebar specific aesthetics */
[data-testid="stSidebar"] {{
    background: #fafafa !important;
    border-right: 1px solid #eee;
}}

.sidebar-logo {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin-bottom: 20px;
    margin-top: -3rem;
    padding: 1rem;
}}
.sidebar-logo img {{
    max-width: 180px;
    drop-shadow: 0px 5px 15px rgba(0,0,0,0.1);
}}
.sidebar-logo-subtitle {{
    margin-top: 0.35rem;
    color: #334155;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0;
    line-height: 1.25;
    text-align: center;
}}

/* Language selector: flag images avoid OS emoji flag rendering */
.language-selector-label {{
    margin: 0 0 0.45rem 0;
    color: var(--text-main);
    font-size: 0.88rem;
    font-weight: 500;
}}
[class*="st-key-lang_"] button {{
    min-width: 0;
    min-height: 2.7rem;
    padding: 0.42rem 0.56rem;
    border: 1px solid #d9dee8 !important;
    background: #fff !important;
    color: var(--text-main) !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
}}
[class*="st-key-lang_"] button p {{
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 0.88rem;
    font-weight: 700;
}}
[class*="st-key-lang_"] button::before {{
    content: "";
    width: 1.72rem;
    height: 1.15rem;
    margin-right: 0.5rem;
    flex: 0 0 auto;
    border-radius: 3px;
    background-position: center;
    background-size: cover;
    box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.16);
}}
.st-key-lang_pt button::before {{
    background-image: url("{FLAG_IMAGE_URIS['pt']}");
}}
.st-key-lang_en button::before {{
    background-image: url("{FLAG_IMAGE_URIS['en']}");
}}
[class*="st-key-lang_"] button:hover:enabled {{
    border-color: var(--primary-red) !important;
    background: #fff7f4 !important;
    box-shadow: 0 6px 16px rgba(255, 64, 17, 0.1) !important;
    transform: translateY(-1px);
}}
[class*="st-key-lang_"] button[kind="primary"] {{
    border-color: var(--primary-red) !important;
    background: #fff4ef !important;
    color: #c53114 !important;
    box-shadow: inset 0 0 0 1px rgba(255, 64, 17, 0.22) !important;
}}

/* Buttons */
button {{
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: var(--font-ui);
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
    font-family: var(--font-ui) !important;
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
    overflow-wrap: anywhere;
    word-break: normal;
}}
[data-testid="stChatMessage"] li {{
    overflow-wrap: anywhere;
    word-break: normal;
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
.st-key-welcome_discover {{
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
    overflow: hidden;
}}
.st-key-welcome_discover::before {{
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
.welcome-discover-title {{
    margin: 0 0 6px;
    font-family: var(--font-display);
    font-size: clamp(1.25rem, 2.6vw, 1.65rem);
    font-weight: 800;
    color: #1f2937;
    line-height: 1.25;
}}
.welcome-discover-text {{
    margin: 0;
    font-size: 0.97rem;
    color: #3a4252;
    line-height: 1.55;
    max-width: 56ch;
}}
.st-key-welcome_discover_cta button {{
    padding: 14px 24px;
    border: none !important;
    border-radius: 14px !important;
    background: linear-gradient(135deg, #ff4011 0%, #ff7e3d 100%) !important;
    color: #fff !important;
    box-shadow: 0 12px 26px -10px rgba(255, 64, 17, 0.6) !important;
    white-space: nowrap;
}}
.st-key-welcome_discover_cta button p {{
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.01em;
}}
.st-key-welcome_discover_cta button::after {{
    content: "\\2192";
    margin-left: 10px;
    font-weight: 800;
}}
.st-key-welcome_discover_cta button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 18px 32px -12px rgba(255, 64, 17, 0.7) !important;
    color: #fff !important;
}}

@media (max-width: 420px) {{
    [class*="st-key-lang_"] button {{
        min-height: 2.55rem;
        padding: 0.4rem 0.42rem;
    }}
    [class*="st-key-lang_"] button p {{
        font-size: 0.82rem;
    }}
    [class*="st-key-lang_"] button::before {{
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
.sidebar-footer-supervisors {{
    color: var(--text-muted);
    font-size: 0.72rem;
    line-height: 1.35;
    margin-top: 0.45rem;
}}
.sidebar-footer-supervisors strong {{
    color: var(--text-main);
    font-weight: 700;
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

/* ============ RESPONSIVE TABLES IN CHAT ============ */
/* Transport/itinerary answers may contain wide tables. On narrow screens
   let them scroll horizontally inside the message instead of breaking the
   page layout. */
[data-testid="stChatMessage"] table {{
    display: block;
    width: fit-content;
    max-width: 100%;
    overflow-x: auto;
    border-collapse: collapse;
}}
[data-testid="stChatMessage"] table th,
[data-testid="stChatMessage"] table td {{
    white-space: nowrap;
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


# Visitor model settings. Presets only fill the base URL field; any
# OpenAI-compatible base URL typed by the visitor works.
LLM_PRESETS: Dict[str, Dict[str, str]] = {
    "openai": {"label": "OpenAI", "base_url": "https://api.openai.com/v1", "model_hint": "gpt-5.4-mini"},
    "azure": {
        "label": "Azure OpenAI",
        "base_url": "",
        "url_hint": "https://<resource>.openai.azure.com/openai/v1/",
        "model_hint": "gpt-5.4-mini",
    },
    "gemini": {
        "label": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_hint": "gemini-2.5-flash",
    },
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "model_hint": "openai/gpt-5.4-mini"},
    "lmstudio": {"label": "LM Studio (local)", "base_url": "http://localhost:1234/v1", "model_hint": "qwen/qwen3.5-9b"},
    "other": {"label": "", "base_url": "", "url_hint": "https://", "model_hint": ""},
}
DEFAULT_LLM_PRESET = "openai"
# Session key of an assistant built on the server provider from config.py.
SERVER_LLM_KEY = "server"


def visitor_llm_settings_enabled() -> bool:
    """Return whether visitors may use their own model in the sidebar."""
    return bool(Config.ALLOW_VISITOR_LLM_SETTINGS)


def llm_preset_options() -> List[str]:
    """Return the presets this deployment can reach; local servers need private hosts."""
    return [key for key in LLM_PRESETS if key != "lmstudio" or Config.ALLOW_PRIVATE_LLM_HOSTS]


def session_llm_connection() -> Optional[LLMConnection]:
    """Return the visitor's model connection for this session, if any."""
    if not visitor_llm_settings_enabled():
        return None
    return st.session_state.get("llm_connection")


def session_llm_ready() -> bool:
    """Return whether this session has a model it can build the assistant with."""
    return session_llm_connection() is not None or LLMFactory.server_provider_configured()


def session_llm_key() -> str:
    """Return a non-reversible key of the model this session should use."""
    connection = session_llm_connection()
    return connection.fingerprint if connection is not None else SERVER_LLM_KEY


def assistant_ready_for_session() -> bool:
    """Return whether the session assistant exists and uses the session model."""
    return bool(
        st.session_state.get("initialized")
        and st.session_state.get("assistant") is not None
        and st.session_state.get("assistant_llm_key") == session_llm_key()
    )


def llm_setup_message() -> str:
    """Explain where to configure a model when none is available."""
    if visitor_llm_settings_enabled():
        return t("llm_setup_prompt")
    return t("llm_server_setup_prompt")


def redact_session_secret(text: str) -> str:
    """Remove the visitor's key from text bound for logs."""
    connection = session_llm_connection()
    if connection is not None and connection.api_key:
        return text.replace(connection.api_key, "[REDACTED]")
    return text


def init_system_state():
    """Initialise session state with secure defaults."""
    defaults = {
        "messages": [],
        "assistant": None,
        "assistant_llm_key": None,
        "llm_connection": None,
        "llm_notice": None,
        "language": "pt",
        "initialized": False,
        "current_page": "chat",
        "startup_resources_attempted": False,
        "startup_resources_ok": None,
        "startup_resources_status": {},
        "transport_db_status": None,
        "startup_auto_init_attempted_key": None,
        "startup_auto_init_error": None,
        "startup_auto_init_in_progress": False,
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


def set_language(language: str) -> None:
    """Switch the UI language within the current session.

    A link would load a new page and start a new session, losing the
    conversation and any visitor model settings; a widget callback keeps them.
    """
    safe_language = language if language in LANGUAGE_OPTIONS else "pt"
    st.session_state.language = safe_language
    st.query_params["lang"] = safe_language
    # The browser keeps a selectbox value as its displayed label. Setting the
    # value again makes Streamlit send the label in the new language;
    # otherwise the next rerun returns the old label ("Outro"), not an option.
    if "llm_preset" in st.session_state:
        st.session_state.llm_preset = st.session_state.llm_preset


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
    """Render the language buttons; flags come from CSS (``st-key-lang_*``)."""
    current_language = st.session_state.get("language", "pt")
    st.markdown(
        f'<p class="language-selector-label">{html.escape(t("language"))}</p>',
        unsafe_allow_html=True,
    )
    columns = st.columns(len(LANGUAGE_OPTIONS))
    for column, (language_key, option) in zip(columns, LANGUAGE_OPTIONS.items()):
        column.button(
            option["label"],
            key=f"lang_{language_key}",
            type="primary" if language_key == current_language else "secondary",
            use_container_width=True,
            disabled=request_locked,
            on_click=set_language,
            args=(language_key,),
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


@st.cache_resource(show_spinner=False)
def _load_shared_startup_resources() -> Dict[str, Any]:
    """Load shared runtime resources once per Streamlit server process.

    This intentionally caches only process-wide resources such as the vector
    store embeddings and generated transport databases: on a fresh hosted
    container it downloads the vector DB and transport release assets into the
    writable runtime directory. Hosted deployments then refresh that data in
    the background (``schedule_runtime_data_refresh``). The conversational
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


def initialize_assistant(show_spinner: bool = True) -> Tuple[bool, Optional[str]]:
    """Build the session assistant with the model this session should use.

    Args:
        show_spinner: Whether to show a spinner while the agents are built.

    Returns:
        Tuple[bool, Optional[str]]: Success flag and a localized error message.
    """
    lang = st.session_state.get("language", "pt")
    if not session_llm_ready():
        st.session_state.initialized = False
        return False, llm_setup_message()

    try:
        from agent.graph import MultiAgentAssistant

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

        init_message = (
            "🤖️ A iniciar o assistente..."
            if lang == "pt"
            else "🤖️ Initializing assistant..."
        )
        if show_spinner:
            with st.spinner(init_message):
                assistant = MultiAgentAssistant(connection=session_llm_connection())
        else:
            assistant = MultiAgentAssistant(connection=session_llm_connection())

        st.session_state.assistant = assistant
        st.session_state.assistant_llm_key = session_llm_key()
        st.session_state.initialized = True
        return True, None
    except Exception:
        logging.getLogger(__name__).error(
            "Assistant initialization failed:\n%s",
            redact_session_secret(traceback.format_exc()),
        )
        st.session_state.assistant = None
        st.session_state.initialized = False
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
        <div class="top-banner">
            <div class="top-banner-container">
                <h1>{t("app_title")}</h1>
                <p>{t("app_subtitle")}</p>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def _apply_llm_preset() -> None:
    """Fill the base URL field with the selected preset."""
    st.session_state.llm_base_url = LLM_PRESETS[st.session_state.llm_preset]["base_url"]


def activate_llm_connection(connection: Optional[LLMConnection]) -> None:
    """Use a model for this session's next messages, keeping the conversation.

    Args:
        connection: Visitor connection, or None for the server default.
    """
    st.session_state.llm_connection = connection
    assistant = st.session_state.get("assistant")
    if assistant is not None:
        assistant.set_llm_connection(connection)
        st.session_state.assistant_llm_key = session_llm_key()


def save_llm_settings() -> None:
    """Validate and test the visitor's model, then use it for this session."""
    try:
        connection = LLMConnection.create(
            base_url=st.session_state.get("llm_base_url", ""),
            model=st.session_state.get("llm_model", ""),
            api_key=st.session_state.get("llm_api_key", ""),
        )
    except ValueError as exc:
        st.session_state.llm_notice = ("error", f"llm_error_{exc}", {})
        return

    with st.spinner(t("llm_checking")):
        failure = check_tool_calling(connection)
    if failure:
        st.session_state.llm_notice = ("error", f"llm_error_{failure}", {})
        return

    activate_llm_connection(connection)
    st.session_state.llm_notice = ("success", "llm_check_ok", {"model": connection.model})


def render_llm_settings(request_locked: bool) -> None:
    """Render the visitor's model settings for this browser session.

    Args:
        request_locked: Whether a request is running and inputs are disabled.
    """
    connection = session_llm_connection()
    server_ready = LLMFactory.server_provider_configured()
    if "llm_preset" not in st.session_state:
        st.session_state.llm_preset = DEFAULT_LLM_PRESET
        st.session_state.llm_base_url = LLM_PRESETS[DEFAULT_LLM_PRESET]["base_url"]

    with st.expander("⚙️ " + t("settings"), expanded=not session_llm_ready()):
        if connection is not None:
            host = urlsplit(connection.base_url).hostname
            st.caption(t("llm_active_custom").format(model=connection.model, host=host))
        elif server_ready:
            st.caption(t("llm_active_server"))
        else:
            st.warning(t("llm_server_unavailable"))

        st.selectbox(
            t("llm_preset"),
            options=llm_preset_options(),
            format_func=lambda key: LLM_PRESETS[key]["label"] or t("llm_preset_other"),
            key="llm_preset",
            on_change=_apply_llm_preset,
            disabled=request_locked,
        )
        preset = LLM_PRESETS[st.session_state.llm_preset]
        st.text_input(
            t("llm_base_url"),
            key="llm_base_url",
            placeholder=preset.get("url_hint", preset["base_url"]),
            disabled=request_locked,
        )
        st.text_input(t("llm_api_key"), key="llm_api_key", type="password", disabled=request_locked)
        st.text_input(
            t("llm_model"),
            key="llm_model",
            placeholder=preset["model_hint"],
            help=t("llm_model_help"),
            disabled=request_locked,
        )
        st.caption(t("llm_key_privacy"))

        if st.button(
            t("llm_save"),
            use_container_width=True,
            type="primary",
            key="llm_save_button",
            disabled=request_locked,
        ):
            save_llm_settings()
            st.rerun()
        if connection is not None and server_ready and st.button(
            t("llm_use_default"),
            use_container_width=True,
            key="llm_default_button",
            disabled=request_locked,
        ):
            activate_llm_connection(None)
            st.session_state.llm_notice = ("success", "llm_default_restored", {})
            st.rerun()

        notice = st.session_state.get("llm_notice")
        if notice:
            kind, message_key, params = notice
            show_notice = st.success if kind == "success" else st.error
            show_notice(t(message_key).format(**params))


def build_sidebar():
    with st.sidebar:
        request_locked = request_capture_locked(
            st.session_state.get("pending_request"),
            st.session_state.get("request_running", False),
            startup_initializing=st.session_state.get(
                "startup_auto_init_in_progress", False
            ),
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

        if visitor_llm_settings_enabled():
            st.divider()
            render_llm_settings(request_locked)
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

        for idx, (icon, label, qt) in enumerate(quick_acts):
            st.button(
                f"{icon} {label}",
                use_container_width=True,
                key=f"sidebar_qact_{idx}",
                disabled=request_locked,
                on_click=queue_pending_request,
                args=(qt,),
            )

        st.divider()

        # Session info
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric(t("messages"), count_user_interactions(st.session_state.messages))
        with col_s2:
            if assistant_ready_for_session():
                st.metric(t("status"), "🟢")
            elif session_llm_ready():
                st.metric(t("status"), "🟡")
            else:
                st.metric(t("status"), "⚪")

        if st.session_state.initialized and hasattr(st.session_state, "assistant") and st.session_state.assistant:
            model_name = getattr(st.session_state.assistant, "model_name", None)
            if model_name:
                st.caption(f"🤖️ {model_name}")

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

        st.markdown(
            f"""
        <div class="sidebar-footer">
            <div class="sidebar-footer-version">{t("footer_version")}</div>
            <div class="sidebar-footer-made">{t("footer_made")}</div>
            <div class="sidebar-footer-supervisors"><strong>{html.escape(t("info_supervisors_label"))}:</strong> {html.escape(t("info_supervisors_names"))}</div>
            <div class="sidebar-footer-made" style="margin-top:2px;">{format_lisbon_now()}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def build_welcome():
    st.markdown(
        f"<h2 style='text-align: center; margin-bottom: 10px;'>{t('welcome_title')}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align: center; color: var(--text-muted); font-size: 1.1rem;'>{t('welcome_intro')}</p>",
        unsafe_allow_html=True,
    )

    # A button, not a link: a link would start a new session (see set_language).
    with st.container(key="welcome_discover"):
        st.markdown(
            f"""
            <span class="welcome-discover-eyebrow">✨ {t('discover_eyebrow')}</span>
            <div class="welcome-discover-title">{t('discover_title')}</div>
            <p class="welcome-discover-text">{t('discover_subtitle')}</p>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            f"ℹ️ {t('discover_cta')}",
            key="welcome_discover_cta",
            on_click=set_navigation_query_params,
            args=("info",),
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
    for i, (ic, lab, qt) in enumerate(examples):
        with cols[i % 3]:
            st.button(
                f"{ic} {lab}",
                use_container_width=True,
                key=f"exq_{i}",
                on_click=queue_pending_request,
                args=(qt,),
            )
    return None


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
        if (
            len(line) > 120
            and (
                re.search(r"^\s*(?:#{1,6}\s+|[-*]\s+|\d+[.)]\s+)", line)
                or "**" in line
                or "](" in line
                or re.search(r"\bhttps?://", line)
            )
        ):
            yield line + suffix
            time.sleep(0.03)
            continue
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
    placeholder.markdown(final_text)
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

    from agent.utils.response_formatter import (
        ensure_transport_time_route_paragraph_breaks,
        ensure_streamlit_standalone_label_blocks,
        repair_split_planner_field_lines,
        restore_initial_pseudo_heading,
    )

    lines = restore_initial_pseudo_heading(text).splitlines()
    output: list[str] = []
    in_fenced_code = False
    field_label_re = re.compile(
        r"^\*\*(?:[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s*)?"
        r"(?:Morada|Address|Localiza(?:ção|cao)|Location|Descrição|Descricao|Description|"
        r"Categoria|Category|Horário|Horario|Hours|Opening hours|Preço|Preco|Price|"
        r"Website|Site oficial|Telefone|Phone|Email|Bilhetes|Tickets|Data/Hora|Date/Time|"
        r"Duração|Duracao|Duration|Características|Caracteristicas|Features):\*\*$",
        flags=re.IGNORECASE,
    )
    for index, line in enumerate(lines):
        stripped = line.strip()
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if stripped.startswith("```"):
            in_fenced_code = not in_fenced_code
            output.append(line)
            continue
        if in_fenced_code:
            output.append(line)
            continue
        if (
            re.match(r"^\*\*[^*\n]{2,180}:?\*\*$", stripped)
            and not field_label_re.match(stripped)
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
    normalized = repair_split_planner_field_lines(
        ensure_streamlit_standalone_label_blocks(
            ensure_transport_time_route_paragraph_breaks("\n".join(output))
        )
    )
    if text.endswith("\n"):
        normalized += "\n"
    return normalized


def clean_response_for_display(text: str, *, rendered_plan: bool = False) -> str:
    """Remove citation artefacts and apply the final render-safe Markdown guard.

    Args:
        text: Final assistant answer.
        rendered_plan: Whether the answer is a plan rendered by the planner's
            deterministic renderer; its Markdown is final, and the guard
            written for free-form answers would rewrite it.
    """
    from agent.utils.response_formatter import (
        final_post_qa_guard,
        final_visual_pass,
        infer_response_language,
    )

    cleaned = re.sub(r"【.*?】", "", text or "")
    cleaned = cleaned.replace("\x00", "").strip()
    if rendered_plan:
        return cleaned
    display_language = ""
    assistant = st.session_state.get("assistant")
    if assistant is not None:
        user_context = getattr(assistant, "state", {}).get("user_context", {})
        if isinstance(user_context, dict):
            display_language = str(user_context.get("language") or "").strip().lower()
    if display_language not in {"pt", "en"}:
        display_language = infer_response_language(
            context_text=cleaned,
            default=st.session_state.get("language", "en"),
        )
    return final_post_qa_guard(
        final_visual_pass(cleaned),
        language=display_language,
        repair_sources=False,
    )


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
    llm_ready: bool,
    assistant_ready: bool,
    session_key: str,
    attempted_key: Optional[str],
    last_error: Optional[str],
) -> bool:
    """Return whether the startup flow should auto-initialize the assistant."""
    if not llm_ready or assistant_ready:
        return False
    if attempted_key == session_key and last_error:
        return False
    return True


def startup_auto_initialization_due() -> bool:
    """Return whether this session's assistant should be built at startup."""
    return should_attempt_startup_auto_initialization(
        llm_ready=session_llm_ready(),
        assistant_ready=assistant_ready_for_session(),
        session_key=session_llm_key(),
        attempted_key=st.session_state.get("startup_auto_init_attempted_key"),
        last_error=st.session_state.get("startup_auto_init_error"),
    )


def should_lock_for_startup_auto_initialization() -> bool:
    """Return whether input controls should be locked for startup auto-init."""
    if st.session_state.get("current_page") != "chat":
        return False
    if st.session_state.get("messages") or st.session_state.get("pending_request"):
        return False
    if st.session_state.get("request_running"):
        return True
    return startup_auto_initialization_due()


def attempt_startup_auto_initialization() -> bool:
    """Initialize the assistant automatically when the session model is ready.

    Returns:
        True when initialization was attempted and the app requested a rerun.
        False when no startup initialization is needed.
    """
    if not startup_auto_initialization_due():
        st.session_state.startup_auto_init_in_progress = False
        return False

    st.session_state.startup_auto_init_in_progress = True
    st.session_state.request_running = True
    spinner_text = (
        "🚀 A preparar o LISBOA para a primeira pergunta..."
        if st.session_state.get("language", "pt") == "pt"
        else "🚀 Preparing LISBOA for the first prompt..."
    )
    try:
        with st.spinner(spinner_text):
            success, error = initialize_assistant(show_spinner=False)
    finally:
        st.session_state.request_running = False
        st.session_state.startup_auto_init_in_progress = False

    st.session_state.startup_auto_init_attempted_key = session_llm_key()
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
    startup_initializing: bool = False,
) -> bool:
    """Return whether the UI should temporarily block new requests."""
    return bool(pending_request) or bool(request_running) or bool(startup_initializing)


def build_user_error_message(error: Exception) -> str:
    """Convert backend exceptions into safe, user-friendly messages.

    When the session uses the server default model and visitors may bring
    their own, persistent failures also point to the sidebar settings.
    """
    lang = st.session_state.get("language", "pt")
    error_str = str(error).lower()

    if "401" in error_str or "unauthorized" in error_str:
        message = (
            "Falha de autenticação. Verifique a configuração da chave e do fornecedor."
            if lang == "pt"
            else "Authentication failed. Check the API key and provider settings."
        )
    elif "rate" in error_str or "limit" in error_str:
        return (
            "O limite de pedidos foi atingido. Aguarde um momento e tente novamente."
            if lang == "pt"
            else "The request limit was reached. Please wait a moment and try again."
        )
    elif "content_filter" in error_str or "responsibleaipolicyviolation" in error_str:
        return (
            "O fornecedor bloqueou temporariamente este pedido. Reformule a pergunta e tente novamente."
            if lang == "pt"
            else "The provider temporarily blocked this request. Please rephrase it and try again."
        )
    elif "timeout" in error_str or "connection" in error_str:
        message = (
            "Não foi possível contactar o fornecedor do modelo. Verifique a ligação ou o servidor local."
            if lang == "pt"
            else "Could not reach the selected model provider. Check your connection or local server."
        )
    else:
        message = t("error_generic")

    if visitor_llm_settings_enabled() and session_llm_connection() is None:
        message = f"{message} {t('llm_switch_hint')}"
    return message


def _progress_key(text: str) -> str:
    """Return an accent-insensitive key for progress message routing."""
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def normalize_progress_message(message: str, language: str) -> str:
    """Return one concise, localized, user-facing progress message."""
    is_pt = language == "pt"
    cleaned = re.sub(r"\s+", " ", str(message or "")).strip().replace("**", "")
    key = _progress_key(cleaned)

    labels = {
        "prepare": "🚀 Preparar LISBOA" if is_pt else "🚀 Preparing LISBOA",
        "interpret": "🧠 Interpretar pedido" if is_pt else "🧠 Understanding request",
        "weather": "🌤️ Verificar tempo" if is_pt else "🌤️ Checking weather",
        "transport": "🚌 Otimizar rota" if is_pt else "🚌 Optimizing route",
        "events": "🎭 Procurar eventos" if is_pt else "🎭 Finding events",
        "places": "📍 Procurar locais" if is_pt else "📍 Finding places",
        "municipal": "🏛️ Consultar dados municipais" if is_pt else "🏛️ Checking municipal data",
        "planner": "🗺️ Organizar plano" if is_pt else "🗺️ Organizing plan",
        "validate": "🔎 Verificar resposta" if is_pt else "🔎 Checking answer",
        "final": "✅ Finalizar resposta" if is_pt else "✅ Finalizing answer",
        "default": "⚡ Preparar resposta" if is_pt else "⚡ Preparing answer",
    }

    if not key:
        return labels["default"]
    if re.search(r"\b(?:qa|validat\w*|validac\w*|validacao|verificar|checking|check|review|revis)\b", key):
        return labels["validate"]
    if re.search(r"\b(?:answer\s+ready|response\s+ready|final(?:\s+answer)?|resposta\s+final|complete|completed|pronta|done)\b", key):
        return labels["final"]
    if re.search(r"\b(?:init|startup|prepar|lisboa)\b", key):
        return labels["prepare"]
    if re.search(r"\b(?:interpret|understand|supervisor|routing|pedido|request)\b", key):
        return labels["interpret"]
    if re.search(r"\b(?:weather|tempo|meteo|forecast|previs|warning|aviso|rain|chuva)\b", key):
        return labels["weather"]
    if re.search(r"\b(?:transport|route|rota|metro|carris|cp|train|comboio|bus|autocarro|tram|horario|schedule|departure|partida)\b", key):
        return labels["transport"]
    if re.search(r"\b(?:event|evento|cultural)\b", key):
        return labels["events"]
    if re.search(r"\b(?:municipal|lisboa aberta|service|servico|dados municipais)\b", key):
        return labels["municipal"]
    if re.search(r"\b(?:research|place|places|local|locais|attraction|atrac|restaurant|restaurante|museum|museu)\b", key):
        return labels["places"]
    if re.search(r"\b(?:planner|plan|planning|itinerary|roteiro|itinerario|organizar|organizing)\b", key):
        return labels["planner"]
    return labels["default"]


def write_progress_step(
    status_container: Any,
    label: str,
    seen_statuses: set[str],
    language: str,
) -> str:
    """Update only the current progress label without keeping a visible history."""
    normalized = normalize_progress_message(label, language)
    seen_statuses.add(normalized)
    status_container.update(label=normalized, state="running", expanded=False)
    return normalized


def run_interaction(
    user_input: str,
    user_message_already_rendered: bool = False,
) -> bool:
    """Run one chat turn and optionally initialise inside the assistant bubble."""
    if not user_message_already_rendered:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

    needs_initialization = not assistant_ready_for_session()

    with st.chat_message("assistant"):
        try:
            initial_status_label = (
                "🚀 Preparar LISBOA"
                if st.session_state.language == "pt"
                else "🚀 Preparing LISBOA"
            )
            if not needs_initialization:
                initial_status_label = (
                    "🧠 Interpretar pedido"
                    if st.session_state.language == "pt"
                    else "🧠 Understanding request"
                )

            status_placeholder = st.empty()
            with status_placeholder.status(initial_status_label, expanded=False) as status:
                seen_statuses: set[str] = set()
                last_status = {"label": ""}

                if needs_initialization:
                    success, error = initialize_assistant(show_spinner=False)
                    if not success:
                        status.update(
                            label="⚠️ Não foi possível preparar o LISBOA."
                            if st.session_state.language == "pt"
                            else "⚠️ Could not prepare LISBOA.",
                            state="error",
                        )
                        error_message = error or t("initialization_failed")
                        assistant_error = f"⚠️ {error_message}"
                        st.error(error_message)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": assistant_error}
                        )
                        return False

                    write_progress_step(
                        status,
                        "🧠 Interpretar pedido"
                        if st.session_state.language == "pt"
                        else "🧠 Understanding request",
                        seen_statuses,
                        st.session_state.language,
                    )

                def on_status(msg: str):
                    normalized = str(msg or "").strip()
                    if not normalized:
                        return

                    if normalized == last_status["label"]:
                        return

                    last_status["label"] = normalized
                    write_progress_step(
                        status,
                        normalized,
                        seen_statuses,
                        st.session_state.language,
                    )

                resp = st.session_state.assistant.chat(
                    user_input,
                    verbose=False,
                    on_status_change=on_status,
                    language=st.session_state.language,
                )
                status.update(
                    label="✅ Resposta pronta!"
                    if st.session_state.language == "pt"
                    else "✅ Answer ready!",
                    state="complete",
                    expanded=False,
                )

            status_placeholder.empty()
            planner = getattr(st.session_state.assistant, "agents", {}).get("planner")
            sanitized = clean_response_for_display(
                resp,
                rendered_plan=getattr(planner, "last_synthesis_path", "") == "brief",
            )
            rendered_response = render_assistant_markdown(sanitized)
            st.session_state.messages.append(
                {"role": "assistant", "content": rendered_response}
            )

        except Exception as error:
            # Log the full traceback to the terminal for debugging, without
            # the visitor's key if a provider error echoes it.
            print(f"\n{'=' * 70}")
            print("ERROR during chat interaction:")
            print(f"  Query: {user_input}")
            print(f"  Error type: {type(error).__name__}")
            print(f"  Error message: {redact_session_secret(str(error))}")
            print(f"{'=' * 70}")
            print(redact_session_secret(traceback.format_exc()), end="")
            print(f"{'=' * 70}\n")

            friendly_message = build_user_error_message(error)
            st.error(f"⚠️ {friendly_message}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ {friendly_message}"}
            )
            return False

    return True


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
                linear-gradient(90deg, rgba(255, 64, 17, 0.10) 0%, rgba(246, 218, 0, 0.15) 30%, 
                rgba(14, 224, 113, 0.10) 70%, rgba(55, 119, 255, 0.10) 100%),
                linear-gradient(180deg, rgba(5, 5, 6, 0.72) 0%, rgba(3, 4, 4, 0.75) 100%),
                var(--info-hero-image, linear-gradient(135deg, #111318, #3b424c));
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
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
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
            border: 1px solid rgba(55, 119, 255, 0.18);
            background: rgba(255, 255, 255, 0.86);
            color: #111827;
            font-size: 0.9rem;
            font-weight: 800;
            text-decoration: none;
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .info-author-link:hover {
            transform: translateY(-2px);
            border-color: rgba(55, 119, 255, 0.24);
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

        .info-author-link-thesis {
            grid-column: 1 / -1;
        }

        .info-author-link-thesis::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23111827' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21.42 10.922a1 1 0 0 0-.019-1.838L12.83 5.18a2 2 0 0 0-1.66 0L2.6 9.08a1 1 0 0 0 0 1.832l8.57 3.908a2 2 0 0 0 1.66 0z'/%3E%3Cpath d='M22 10v6'/%3E%3Cpath d='M6 12.5V16a6 3 0 0 0 12 0v-3.5'/%3E%3C/svg%3E");
        }

        .info-author-link-article::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23111827' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z'/%3E%3Cpath d='M14 2v4a2 2 0 0 0 2 2h4'/%3E%3Cpath d='M10 9H8'/%3E%3Cpath d='M16 13H8'/%3E%3Cpath d='M16 17H8'/%3E%3C/svg%3E");
        }

        .info-author-link-doi::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23111827' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect width='20' height='5' x='2' y='3' rx='1'/%3E%3Cpath d='M4 8v11a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8'/%3E%3Cpath d='M10 12h4'/%3E%3C/svg%3E");
        }

        .info-author-publication {
            color: #a8320f;
            font-weight: 700;
            text-decoration: none;
        }

        .info-author-publication:hover { text-decoration: underline; }

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
        f'<p class="info-system-desc">{md_to_html(t("info_architecture_desc"))}</p>'
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
        f'<span class="info-author-secondary">{html.escape(t("info_author_year"))}</span><br>'
        f'<span class="info-author-secondary"><strong>{html.escape(t("info_supervisors_label"))}:</strong> '
        f'{html.escape(t("info_supervisors_names"))}</span>'
        f'<span class="info-author-secondary"><strong>{html.escape(t("info_publication_label"))}:</strong> '
        f'<a class="info-author-publication" href="{ARTICLE_DOI_URL}" target="_blank" rel="noopener noreferrer">'
        '<em>Results in Engineering</em>, 32 (2026), 113369</a></span></p>'
        '</div>'
        '<div class="info-author-links">'
        f'<a class="info-author-link info-author-link-thesis" href="{THESIS_URL}" target="_blank" rel="noopener noreferrer"><span>{html.escape(t("info_author_thesis"))}</span></a>'
        f'<a class="info-author-link info-author-link-article" href="{ARTICLE_DOI_URL}" target="_blank" rel="noopener noreferrer"><span>{html.escape(t("info_author_article"))}</span></a>'
        f'<a class="info-author-link info-author-link-doi" href="{SOFTWARE_DOI_URL}" target="_blank" rel="noopener noreferrer"><span>{html.escape(t("info_author_software"))}</span></a>'
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
    # Hosted only, at most once per LISBOA_DATA_REFRESH_HOURS, in the background.
    schedule_runtime_data_refresh()

    st.session_state.startup_auto_init_in_progress = (
        should_lock_for_startup_auto_initialization()
    )

    build_sidebar()

    if st.session_state.current_page == "info":
        st.session_state.startup_auto_init_in_progress = False
        run_info_page()
        return

    pending = st.session_state.get("pending_request")
    request_running = st.session_state.get("request_running", False)
    request_locked = request_capture_locked(
        pending,
        request_running,
        startup_initializing=st.session_state.get("startup_auto_init_in_progress", False),
    )

    if in_text := st.chat_input(t("chat_placeholder"), disabled=request_locked):
        queue_pending_request(in_text)
        st.rerun()

    pending = st.session_state.get("pending_request")
    request_running = st.session_state.get("request_running", False)
    show_landing = not st.session_state.messages and not pending and not request_running

    banner_slot = st.empty()
    if show_landing:
        with banner_slot.container():
            display_banner()
    else:
        banner_slot.empty()

    if show_landing:
        attempt_startup_auto_initialization()

    request_locked = request_capture_locked(
        pending,
        request_running,
        startup_initializing=st.session_state.get("startup_auto_init_in_progress", False),
    )

    # Stage 1: Capture a new request from quick-action, chat input, or welcome
    # button. Queue it and append the user turn immediately, then rerun once so
    # the sidebar counter and chat history both reflect the new turn before the
    # assistant call starts.
    welcome_request = None
    chat_request = None

    chat_history_slot = st.container()
    with chat_history_slot:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                content = msg["content"]
                if msg["role"] == "assistant":
                    content = normalize_streamlit_chat_markdown(content)
                st.markdown(content)

    welcome_slot = st.empty()
    if show_landing and not request_locked:
        with welcome_slot.container():
            welcome_request = build_welcome()
    else:
        welcome_slot.empty()

    new_request = select_new_request(
        sidebar_request=None,
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
        already_appended = bool(st.session_state.pop("pending_request_user_appended", False))
        st.session_state.pop("pending_request", None)
        try:
            with chat_history_slot:
                run_interaction(
                    pending,
                    user_message_already_rendered=already_appended,
                )
        finally:
            st.session_state.request_running = False
        # Trigger one final rerun so the sidebar counter picks up the assistant
        # turn immediately instead of waiting for the next user action.
        st.rerun()

    if not assistant_ready_for_session():
        auto_init_error = st.session_state.get("startup_auto_init_error")
        attempted_key = st.session_state.get("startup_auto_init_attempted_key")
        if not session_llm_ready():
            # Once a turn has run, its reply already carries this prompt.
            if not st.session_state.messages:
                st.info(llm_setup_message())
        elif auto_init_error and attempted_key == session_llm_key():
            st.error(auto_init_error)
        else:
            st.info(t("llm_ready_prompt"))

if __name__ == "__main__":
    main()
