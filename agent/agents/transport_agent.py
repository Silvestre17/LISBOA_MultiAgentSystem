# ==========================================================================
# Master Thesis - Transport Agent
#   - André Filipe Gomes Silvestre, 20240502
#
#   Specialized agent for transport-related queries.
#   Handles metro, bus, train, and multi-modal routing.
#   Uses BaseAgent.execute_react_loop() for tool execution.
# ==========================================================================

import re
import unicodedata
import uuid
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from agent.agents.base import BaseAgent
from agent.prompts.transport import get_transport_prompt
from agent.utils.geographic_scope import (
    AML_MUNICIPALITY_NAMES,
    build_geographic_out_of_scope_response,
    route_mentions_outside_aml,
)
from agent.utils.langsmith_tracing import traceable
from agent.state import AgentState
from agent.utils.langgraph_compat import ToolNode
from agent.utils.response_formatter import (
    finalize_worker_response,
    has_source_line,
    infer_response_language,
    normalize_cp_no_more_trains_message,
    operators_from_tool_names,
    preserve_contextual_destination_name,
    rebuild_transport_source_line,
)


_CP_LONG_DISTANCE_DESTINATION_RE = re.compile(
    r"\b(?:porto|campanha|campanh[aã]|sao bento|são bento|coimbra|aveiro|braga|guimaraes|guimar[aã]es|faro|algarve)\b",
    re.IGNORECASE,
)
_CP_AML_ROUTE_ANCHORS = {
    "alges",
    "algueirao",
    "amadora",
    "azambuja",
    "barreiro",
    "belem",
    "benfica",
    "cacem",
    "cais do sodre",
    "campolide",
    "carcavelos",
    "cascais",
    "entrecampos",
    "estoril",
    "oeiras",
    "oriente",
    "queluz",
    "rossio",
    "santa apolonia",
    "santo amaro",
    "sao pedro do estoril",
    "sete rios",
    "sintra",
}


def _normalize_token(text: str) -> str:
    """Normalizes station and direction tokens for robust matching."""
    import unicodedata

    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    return normalized.lower().strip()


def _expand_route_address_abbreviations(text: str) -> str:
    """Expand common street abbreviations before regex endpoint extraction."""
    value = str(text or "")
    replacements = (
        (r"\bR\.\s+", "Rua "),
        (r"\bAv\.\s+", "Avenida "),
        (r"\bTv\.\s+", "Travessa "),
        (r"\bLgo\.\s+", "Largo "),
        (r"\bPç\.\s+", "Praça "),
        (r"\bPrac\.\s+", "Praça "),
        (r"\bEstr\.\s+", "Estrada "),
        (r"\bAl\.\s+", "Alameda "),
    )
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    return value


def _generic_service_area_endpoint(endpoint: str) -> str:
    """Return an area reference for generic service queries such as veterinario em Alges."""
    value = re.sub(r"\s+", " ", str(endpoint or "")).strip(" .,:;?!")
    if not value:
        return ""

    match = re.match(
        r"^(?:o|a|os|as|um|uma|uns|umas|the|a|an)?\s*"
        r"(?P<service>"
        r"veterin[áa]rios?|veterin[áa]rias?|veterinary|"
        r"cl[ií]nicas?\s+veterin[áa]rias?|hospital\s+veterin[áa]rio|"
        r"farm[áa]cias?|pharmac(?:y|ies)|restaurantes?|restaurants?|"
        r"caf[eé]s?|cafes?|tabernas?|bares?|bars?|lojas?|stores?|shops?"
        r")\s+"
        r"(?:em|no|na|nos|nas|perto\s+de|junto\s+de|in|near|around)\s+"
        r"(?P<area>.+)$",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""

    area = re.sub(r"\s+", " ", match.group("area")).strip(" .,:;?!")
    area = re.sub(
        r"\s+(?:de\s+metro|de\s+autocarro|de\s+comboio|by\s+metro|by\s+bus|by\s+train)\b.*$",
        "",
        area,
        flags=re.IGNORECASE,
    ).strip(" .,:;?!")
    if len(area) < 2:
        return ""
    return area


def _ambiguity_preamble_is_no_clear_match(text: str) -> bool:
    """Return whether an ambiguity note is a no-match prompt rather than real alternatives."""
    normalized = _normalize_token(text)
    return bool(
        re.search(
            r"\b(?:preciso de confirmar|i need to confirm|nao encontrei uma correspondencia clara|could not find a clear match)\b",
            normalized,
        )
    )


def _insert_before_source_footer(text: str, note: str) -> str:
    """Insert a short limitation note before the final source footer."""
    if not text or not note:
        return text
    marker_re = re.compile(r"\n\n(?=📌\s+\*\*(?:Fonte|Source):)", flags=re.IGNORECASE)
    if marker_re.search(text):
        return marker_re.sub(f"\n\n{note}\n\n", text, count=1)
    return f"{text.rstrip()}\n\n{note}"


def _append_generic_service_area_note(
    response: str,
    raw_destination: str,
    area_destination: str,
    language: str,
) -> str:
    """Clarify that a generic service route uses the area as the destination anchor."""
    if not response or not raw_destination or not area_destination:
        return response
    raw_key = _normalize_token(raw_destination)
    area_key = _normalize_token(area_destination)
    if not raw_key or not area_key or raw_key == area_key:
        return response
    note = (
        f"⚠️ **Nota:** como não há uma morada/nome específico confirmado para **{raw_destination}**, "
        f"usei **{area_destination}** como ponto de referência de chegada."
        if language == "pt"
        else f"⚠️ **Note:** because no specific confirmed address/name was available for **{raw_destination}**, "
        f"I used **{area_destination}** as the destination reference."
    )
    if _normalize_token(note) in _normalize_token(response):
        return response
    direct_re = re.compile(
        r"(?m)^(?P<line>✅\s+\*\*(?:Resposta direta|Direct answer):\*\*\s+[^\n]+)$",
        flags=re.IGNORECASE,
    )
    direct_match = direct_re.search(response)
    if direct_match:
        direct_note = (
            f" Como não há uma morada/nome específico confirmado para **{raw_destination}**, "
            f"usei **{area_destination}** como ponto de referência de chegada."
            if language == "pt"
            else f" Because no specific confirmed address/name was available for **{raw_destination}**, "
            f"I used **{area_destination}** as the destination reference."
        )
        return direct_re.sub(lambda match: f"{match.group('line')}{direct_note}", response, count=1)
    return _insert_before_source_footer(response, note)


def _strip_endpoint_mode_clauses(value: str) -> str:
    """Remove transport-mode constraints accidentally captured as place text."""
    cleaned = str(value or "").strip()
    mode_terms = (
        r"metro|autocarros?|bus(?:es)?|el[eé]tricos?|trams?|comboios?|trains?|cp|carris"
    )
    quality_terms = (
        r"caminhad|andar|walk(?:ing)?|transfer[eê]ncias?|transbordos?|escadas?|stairs?|"
        r"acess[ií]vel|accessible|cadeira\s+de\s+rodas|wheelchair|mobilidade|chuva|"
        r"chover|rain(?:y|s|ing)?|bagagem|luggage|malas?|idosos?|elderly|"
        r"crian[cç]as?|children|kids"
    )
    cleanup_patterns = (
        rf"\s+(?:que|qual|which|what)\s+(?:{mode_terms})\b.*$",
        rf"\s+(?:e|and)\s+(?:que|qual|which|what)\s+(?:{mode_terms})\b.*$",
        rf"\s+(?:sem|without|no)\s+(?:{mode_terms})\b.*$",
        rf"\s+(?:s[oó]\s+(?:de\s+)?|only\s+)(?:{mode_terms})\b.*$",
        rf"\s+(?:com|using|by)\s+(?:{mode_terms})\b.*$",
        rf"\s+(?:de|por)\s+(?:{mode_terms})\b.*$",
        r"\s+(?:quanto\s+tempo|a\s+que\s+horas|quando|how\s+long|when)\b.*$",
        rf"\s+(?:com|sem|with|without)\s+(?=[^,;.!?]*(?:{quality_terms})).*$",
        rf"\s+(?:e|and)\s+(?:com|sem|with|without)\s+(?=[^,;.!?]*(?:{quality_terms})).*$",
        rf"\s+(?:e\s+)?(?:se|if)\s+(?=[^,;.!?]*(?:{quality_terms})).*$",
        rf"\s+(?:e|and)\s+(?:se|if)\s+(?=[^,;.!?]*(?:{quality_terms})).*$",
        rf"\s+(?:em\s+caso\s+de|caso|in\s+case\s+of)\s+(?=[^,;.!?]*(?:{quality_terms})).*$",
        rf"\s+(?:à|a|na|no|in)\s+(?=[^,;.!?]*(?:{quality_terms})).*$",
    )
    for pattern in cleanup_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip(" .?!,;:")
    return cleaned


def _clean_query_fragment(part: str) -> str:
    """Cleans station and stop fragments parsed from transport questions."""
    alias_map = {
        "amoreiras shopping": "Amoreiras Shopping Center",
        "amoreiras shopping center": "Amoreiras Shopping Center",
        "amoreiras centro comercial": "Amoreiras Shopping Center",
        "airport": "Aeroporto",
        "centro comercial amoreiras": "Amoreiras Shopping Center",
        "lisbon airport": "Aeroporto",
        "airport terminal": "Aeroporto",
        "chiado": "Baixa-Chiado",
    }

    part = _strip_endpoint_mode_clauses(part.strip(" .?!,;:"))
    located_at_match = re.match(
        r"^(?P<label>.+?)\s+que\s+fica\s+(?:na|no|em|at)\s+(?P<tail>.+)$",
        part,
        flags=re.IGNORECASE,
    )
    if located_at_match:
        label = located_at_match.group("label").strip(" .?!,;:")
        tail = located_at_match.group("tail").strip(" .?!,;:")
        tail_norm = _normalize_token(tail)
        venue_label = re.search(
            r"\b(?:centro\s+comercial|shopping|farm[aá]cia|loja|restaurante|caf[eé]|hotel|"
            r"museu|biblioteca|hospital|universidade|faculdade|escola|teatro|cinema|aeroporto|mercado)\b",
            label,
            flags=re.IGNORECASE,
        )
        exact_address = re.search(r"\b\d+[A-Za-z]?\b|\b\d{4}-\d{3}\b", tail)
        if venue_label and exact_address:
            part = f"{label}, {tail}"
        elif venue_label:
            part = label
        elif re.search(
            r"\b(?:rua|avenida|av|av\.|largo|praca|praça|travessa|calçada|calcada|"
            r"estrada|alameda|campo|campus|n[oº]?|numero|número)\b|\b\d{4}-\d{3}\b",
            tail_norm,
        ):
            part = tail
        else:
            part = label
    part = re.sub(
        r"\s+(?:para|to)\s+(?:chegar|chegares|chegarmos|estar|arrive|get\s+there)\b.*$",
        "",
        part,
        flags=re.IGNORECASE,
    ).strip(" .?!,;:")
    part = re.sub(
        r"\s+(?:que|qual|which|what)\s+"
        r"(?:autocarro|autocarros|bus|buses|metro|comboio|comboios|train|trains|"
        r"el[eé]trico|el[eé]tricos|tram|trams)\b.*$",
        "",
        part,
        flags=re.IGNORECASE,
    ).strip(" .?!,;:")
    part = re.sub(
        r"\s+(?:quanto\s+tempo|a\s+que\s+horas|quando|how\s+long|when)\b.*$",
        "",
        part,
        flags=re.IGNORECASE,
    ).strip(" .?!,;:")
    part = re.sub(
        r"\s+(?:e|and)\s+(?:ir|seguir|chegar|go|get|travel)\s*$",
        "",
        part,
        flags=re.IGNORECASE,
    ).strip(" .?!,;:")
    part = re.sub(
        r"^(?:e\s+se\s+(?:for|fosse)\s+)?(?:apenas|s[oó]|only|just)\s+(?:de\s+|from\s+)?",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"\b(agora|pfv|por favor|sff|please|now|right now|já|ja|mesmo|pff|only|just|apenas|s[oó])\b.*$",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"\b(without(?:\s+taking)?\s+(?:a\s+)?(?:bus|buses|tram|trams)|sem\s+(?:autocarro|autocarros|eletrico|el[eé]trico|eletricos|el[eé]tricos)|(?:bus|tram)\s+only|s[oó]\s+de\s+(?:autocarro|autocarros|eletrico|el[eé]trico|eletricos|el[eé]tricos))\b.*$",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"\b(?:sem\s+complica(?:[çc](?:ão|ao|ões|oes))|qual(?:\s+[ée])?\s+a?\s*melhor\s+(?:forma|rota|caminho)|what(?:'s| is)\s+the\s+best\s+(?:way|route)|best\s+(?:way|route)|o\s+metro\s+serve\s+bem|serve\s+bem)\b.*$",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"\b(?:e|and)\s+(?:o\s+que\s+muda|what\s+changes?|what\s+is\s+different|como\s+muda)\b.*$",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"\s+(?:e|and)\s+(?:como|depois|a\s+seguir|de\s+seguida|then|after|next)\b.*$",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"\s+(?:e|and)\s+(?:h[aá]|existem|tem|t[eê]m|are\s+there|is\s+there)\s+"
        r"(?:atrasos?|perturba[cç][oõ]es|avisos?|alertas?|delays?|disruptions?|warnings?|alerts?)\b.*$",
        "",
        part,
        flags=re.IGNORECASE,
    )
    route_mode_fragment = (
        r"(?:metro|bus|buses|autocarro|autocarros|tram|trams|el[eé]trico|el[eé]tricos|"
        r"comboio|comboios|train|trains)"
    )
    part = re.sub(
        rf"^(?:{route_mode_fragment})(?:\s+(?:e/ou|e|ou|or|and|vs|versus)\s+"
        rf"(?:(?:uma|um|one)\s+(?:op[cç][aã]o|option)\s+)?(?:de\s+|by\s+)?"
        rf"{route_mode_fragment})+\s+(?:dos|das|do|da|de|from)\s+",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"\b(?:de metro|de autocarro|de comboio|de elétrico|de eletrico|(?:by|using|via)(?:\s+the)?\s+(?:metro|bus|tram|train))\b",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"\b(?:or\s+(?:bus|tram|train|metro)|ou\s+(?:autocarro|el[eé]trico|comboio|metro))\b.*$",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"\b(?:de\s+transportes?\s+p[úu]blicos?|por\s+transportes?\s+p[úu]blicos?|by\s+public\s+transport|using\s+public\s+transport)\b",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"^(?:transporte\s+p[úu]blico|transportes\s+p[úu]blicos|public\s+transport)\s+(?:do|da|de|from)\s+",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"^(?:uma\s+|um\s+)?(?:alternativa|op[cç][aã]o|outra\s+op[cç][aã]o|"
        r"meios?\s+de\s+transporte|transportes?|transporte)\s+"
        r"(?:do|da|dos|das|de|from)\s+",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"^(?:autocarro|autocarros|bus|buses)\s+(?:da|de)\s+carris\s+metropolitana\s+(?:de|do|da)\s+",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"^(?:metro|bus|tram|train|comboio|autocarro|el[eé]trico)\s+(?:de|do|da|dos|das|from)\s+",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"^(?:e\s+se\s+(?:for|fosse)\s+)?(?:apenas|s[oó]|only|just)\s+(?:de\s+|from\s+)?",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"\b(metro station|station|esta[cç][aã]o|stop|paragem|terminal)\b",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(r"\b(at[eé]|ate)\s*$", "", part, flags=re.IGNORECASE)
    part = re.sub(
        r"^(?:ir|vou|quero ir|preciso ir|preciso de ir|go|get|travel|take)\s+(?:de\s+|do\s+|da\s+|from\s+)?",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"^(?:para|ao|a|à|to)\s+",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"\b(?:hoje|amanh[ãa]|esta\s+tarde|hoje\s+[àa]\s+tarde|[àa]\s+tarde|today|tomorrow|this\s+afternoon)\b.*$",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(
        r"\b(?:how\s+do\s+i\s+get\s+there|how\s+to\s+get\s+there|como\s+l[aá]\s+chego|como\s+chego\s+l[aá])\b.*$",
        "",
        part,
        flags=re.IGNORECASE,
    )
    part = re.sub(r"^(o|a|os|as|the)\s+", "", part, flags=re.IGNORECASE)
    part = part.strip(" .?!,;:")
    try:
        from tools.location_resolver import clean_location_query_fragment

        part = clean_location_query_fragment(part) or part
    except Exception:
        pass
    normalized = _normalize_token(part)
    return alias_map.get(normalized, part)


def _extract_coordinates(query: str) -> Optional[Tuple[float, float]]:
    """Extracts latitude/longitude pairs from a transport query when available."""
    match = re.search(
        r"(?P<latitude>-?\d{1,2}\.\d+)\s*[,;/]\s*(?P<longitude>-?\d{1,3}\.\d+)",
        query,
    )
    if not match:
        return None

    try:
        return float(match.group("latitude")), float(match.group("longitude"))
    except (TypeError, ValueError):
        return None


def _canonicalize_route_code(raw_route: str) -> Optional[str]:
    """Normalizes Carris route codes such as `28 E` -> `28E`."""
    cleaned = re.sub(r"\s+", "", str(raw_route or "").upper())
    if re.fullmatch(r"\d{1,4}[A-Z]?", cleaned):
        return cleaned
    return None


def _extract_route_code(query: str) -> Optional[str]:
    """Extracts a Carris bus/tram route code from a natural-language query."""
    patterns = [
        r"\b(?:route|linha|line|tram|trams|el[eé]trico|eletrico|bus|buses|autocarro|autocarros)\s*(?:number|n[uú]mero|n[oº]?|#)?\s*(?P<route>\d{1,4}\s*[A-Za-z]?)\b",
        r"\b(?P<route>\d{1,4}\s*E)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            route_code = _canonicalize_route_code(match.group("route"))
            if route_code:
                return route_code
    return None


def _resolve_named_candidate(
    fragment: str,
    candidate_map: Dict[str, str],
    minimum_score: float = 0.78,
) -> Optional[str]:
    """Resolves a free-form place name against a candidate map with fuzzy matching."""
    normalized_fragment = _normalize_token(fragment)
    if not normalized_fragment:
        return None

    if normalized_fragment in candidate_map:
        return candidate_map[normalized_fragment]

    best_value = None
    best_score = 0.0
    fragment_tokens = set(normalized_fragment.split())
    for candidate_key, canonical in candidate_map.items():
        candidate_tokens = set(candidate_key.split())
        score = SequenceMatcher(None, normalized_fragment, candidate_key).ratio()
        if normalized_fragment in candidate_key or candidate_key in normalized_fragment:
            score += 0.12
        if fragment_tokens and candidate_tokens:
            overlap = len(fragment_tokens & candidate_tokens) / max(
                len(fragment_tokens), len(candidate_tokens)
            )
            score += overlap * 0.25
        if score > best_score:
            best_score = score
            best_value = canonical

    return best_value if best_score >= minimum_score else None


@lru_cache(maxsize=1)
def _get_metro_station_name_map() -> Dict[str, str]:
    """Builds a normalized alias map for Metro station names."""
    from tools.metrolisboa_api import METRO_STATION_IDS, load_metro_stations

    stations_by_code: Dict[str, str] = {}
    for station in load_metro_stations():
        stop_id = str(station.get("stop_id", "")).strip()
        stop_name = str(station.get("stop_name", "")).strip()
        if stop_id and stop_name:
            stations_by_code[stop_id] = stop_name

    alias_map: Dict[str, str] = {}
    for alias, station_id in METRO_STATION_IDS.items():
        canonical = stations_by_code.get(station_id) or alias.title()
        alias_map[_normalize_token(alias)] = canonical

    for canonical in stations_by_code.values():
        alias_map[_normalize_token(canonical)] = canonical

    canonical_overrides = {
        "baixa chiado": "Baixa-Chiado",
        "baixa/chiado": "Baixa-Chiado",
    }
    for alias, canonical in canonical_overrides.items():
        alias_map[_normalize_token(alias)] = canonical

    return alias_map


def _resolve_metro_station_name(fragment: str) -> str:
    """Resolves a Metro station fragment to the best canonical station name."""
    cleaned = _clean_query_fragment(fragment)
    resolved = _resolve_named_candidate(cleaned, _get_metro_station_name_map())
    return resolved or cleaned


@lru_cache(maxsize=1)
def _get_cp_station_name_map() -> Dict[str, str]:
    """Builds a normalized alias map for CP AML station names."""
    from tools.cp_api import CP_KEY_STATIONS, load_cp_aml_stations

    alias_map: Dict[str, str] = {}
    for key, info in CP_KEY_STATIONS.items():
        canonical = str(info.get("name") or key.replace("_", " ").title()).strip()
        alias_map[_normalize_token(canonical)] = canonical
        alias_map[_normalize_token(key.replace("_", " "))] = canonical

    for station in load_cp_aml_stations().values():
        canonical = str(station.get("name") or "").strip()
        if canonical:
            alias_map[_normalize_token(canonical)] = canonical

    extra_aliases = {
        "cais do sodre": "Cais do Sodré",
        "oriente": "Oriente",
        "lisboa oriente": "Oriente",
        "lisboa - oriente": "Oriente",
        "santa apolonia": "Santa Apolónia",
        "belem": "Belém",
        "sete rios": "Sete Rios",
        "sete-rios": "Sete Rios",
    }
    for alias, canonical in extra_aliases.items():
        alias_map[_normalize_token(alias)] = canonical

    return alias_map


def _resolve_cp_station_name(fragment: str) -> str:
    """Resolves a CP station fragment to the best canonical AML station name."""
    cleaned = _clean_query_fragment(fragment)
    resolved = _resolve_named_candidate(cleaned, _get_cp_station_name_map(), minimum_score=0.74)
    return resolved or cleaned


def _extract_metro_line_id(query: str) -> Optional[str]:
    """Extracts the canonical Lisbon Metro line ID from a user query."""
    normalized = _normalize_token(query)
    aliases = {
        "yellow line": "amarela",
        "linha amarela": "amarela",
        "amarela": "amarela",
        "yellow": "amarela",
        "blue line": "azul",
        "linha azul": "azul",
        "azul": "azul",
        "blue": "azul",
        "green line": "verde",
        "linha verde": "verde",
        "verde": "verde",
        "green": "verde",
        "red line": "vermelha",
        "linha vermelha": "vermelha",
        "vermelha": "vermelha",
        "red": "vermelha",
    }
    for alias, line_id in aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return line_id
    return None


def _metro_line_emoji(line_id: str) -> str:
    """Returns the official color emoji for a Metro de Lisboa line."""
    return {
        "amarela": "🟡",
        "azul": "🔵",
        "verde": "🟢",
        "vermelha": "🔴",
    }.get(line_id, "🚇")


def _format_metro_line_set(line_value: str, language: str = "pt") -> str:
    """Formats one or more Metro line identifiers with all relevant colours."""
    if not line_value:
        return ""

    line_labels = {
        "amarela": ("Amarela", "Yellow"),
        "azul": ("Azul", "Blue"),
        "verde": ("Verde", "Green"),
        "vermelha": ("Vermelha", "Red"),
    }
    normalized = _normalize_token(line_value)
    line_ids = [
        line_id
        for line_id in ("amarela", "azul", "verde", "vermelha")
        if re.search(rf"\b{line_id}\b", normalized)
    ]
    if not line_ids:
        return line_value.title()

    emojis = "".join(_metro_line_emoji(line_id) for line_id in line_ids)
    label_index = 0 if language == "pt" else 1
    names = [line_labels[line_id][label_index] for line_id in line_ids]
    if len(names) == 1:
        noun = "Linha" if language == "pt" else "Line"
        return f"{emojis} {noun} {names[0]}"

    if language == "pt":
        joined_names = " e ".join(names) if len(names) == 2 else ", ".join(names[:-1]) + f" e {names[-1]}"
        return f"{emojis} Linhas {joined_names}"

    joined_names = " and ".join(names) if len(names) == 2 else ", ".join(names[:-1]) + f" and {names[-1]}"
    return f"{emojis} {joined_names} Lines"


def _query_requests_metro_line_wait_times(query: str) -> bool:
    """Detects whole-line or all-station Metro wait-time questions."""
    line_id = _extract_metro_line_id(query)
    if not line_id:
        return False

    normalized = _normalize_token(query)
    has_wait_intent = bool(
        re.search(
            r"\b(?:wait(?:ing)?\s+times?|tempo(?:s)?\s+de\s+espera|proximos?\s+metros?|next\s+metros?|next\s+trains?)\b",
            normalized,
        )
        or re.search(r"\bespera\b", normalized)
    )
    has_line_scope = bool(
        re.search(
            r"\b(?:toda\s+a\s+linha|linha\s+toda|em\s+toda\s+a\s+linha|todas?\s+as\s+estacoes|all\s+stations|entire\s+line|whole\s+line|across)\b",
            normalized,
        )
        or re.search(r"\b(?:linha|line)\b", normalized)
    )
    return has_wait_intent and has_line_scope


def _query_requests_compact_metro_wait_summary(query: str) -> bool:
    """Return whether a line wait-time request asks for a compact/key-station summary."""
    normalized = _normalize_token(query)
    return bool(
        re.search(
            r"\b(?:resumo\s+curto|resumo|sintese|síntese|compacto|curto|short\s+summary|brief|"
            r"principais|main|key|hubs?|interchanges?|correspondencias?|correspondências?)\b",
            normalized,
            flags=re.IGNORECASE,
        )
        and re.search(
            r"\b(?:tempos?\s+de\s+espera|wait(?:ing)?\s+times?|proximos?\s+metros?|next\s+metros?|linha|line)\b",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def _build_metro_tool_spec(user_message: str) -> Optional[Dict[str, Any]]:
    """Maps natural-language Lisbon Metro queries to deterministic tool specs."""
    query = user_message.strip()
    query_lower = query.lower()
    line_id = _extract_metro_line_id(query)
    has_metro_context = "metro" in query_lower or line_id is not None

    if has_metro_context and not _query_has_wait_departure_intent(query) and (
        _query_has_status_intent(query)
        or re.search(r"\b(status|estado|working|a funcionar|service)\b", query_lower)
    ):
        return {"name": "get_metro_status", "args": {}}

    # Line-specific wait-time / all-stations-on-line follow-ups (PT + EN)
    if line_id and (
        _query_requests_metro_line_wait_times(query)
        or re.search(
            r"\b(?:wait\s+times?|entire|whole|all\s+stations|linha\s+toda|toda\s+a\s+linha|todas?\s+as\s+esta[cç][oõ]es)\b",
            query_lower,
        )
    ):
        return {"name": "get_metro_line_wait_times", "args": {"line": line_id}}

    # Wait/departure intent across the whole network without a specific line:
    # defer to the LLM ReAct loop so it can fan out to all four lines per the
    # transport prompt instructions, instead of falling into the station-list path.
    if has_metro_context and not line_id and _query_has_wait_departure_intent(query):
        return None

    if has_metro_context and re.search(
        r"\b(all|list|show|every|what|which|todas?|listar|quais)\b.*\b(stations|esta[cç][õo]es|estacoes)\b|\bmetro\s+(?:stations|esta[cç][õo]es|estacoes)\b|\b(stations|esta[cç][õo]es|estacoes)\b.*\bmetro\b",
        query_lower,
    ):
        return {"name": "get_all_metro_stations", "args": {}}

    if line_id and re.search(
        r"\b(frequency|headway|how often|interval|intervalo|frequ[eê]ncia|de quanto em quanto)\b",
        query_lower,
    ):
        language = infer_response_language(user_query=query, default="en")
        return {
            "name": "get_metro_frequency",
            "args": {"line": line_id, "day_type": "weekday", "language": language},
        }

    if "metro" in query_lower and re.search(
        r"\b(nearest|closest|mais perto|mais próxima|mais proxima)\b",
        query_lower,
    ):
        coordinates = _extract_coordinates(query)
        if coordinates:
            return {
                "name": "find_nearest_metro",
                "args": {"latitude": coordinates[0], "longitude": coordinates[1]},
            }

        near_match = re.search(
            r"\b(?:near|to|for|perto de|junto de|ao p[eé] de)\s+(?P<location>.+?)(?:[\?\!\.,;]|$)",
            query,
            flags=re.IGNORECASE,
        )
        if near_match:
            return {
                "name": "find_nearest_metro",
                "args": {"near_location_name": _clean_query_fragment(near_match.group("location"))},
            }

    return None


def _build_carris_urban_tool_spec(user_message: str) -> Optional[Dict[str, Any]]:
    """Maps natural-language Carris urban bus/tram queries to deterministic tool specs."""
    query = user_message.strip()
    query_lower = query.lower()
    endpoints = _extract_route_endpoints(query)
    route_code = _extract_route_code(query)
    has_train_context = bool(re.search(r"\b(cp|comboio|comboios|train|trains)\b", query_lower))

    if has_train_context or _looks_like_carris_metropolitana_query(query, endpoints=endpoints):
        return None

    stop_id_match = re.search(r"\b(?:stop|paragem)\s+(?P<stop_id>\d{2,})\b", query_lower)

    stop_search_patterns = [
        r"(?:search|find|lookup|show)\s+(?:carris\s+)?stops?\s+(?:for|near|named)\s+(?P<term>.+?)(?:[\?\!\.,;]|$)",
        r"(?:what|which)\s+(?:bus|tram|carris)?\s*stops?\s+(?:are\s+)?(?:for|near|around|named)\s+(?P<term>.+?)(?:[\?\!\.,;]|$)",
        r"(?:paragens?|stops?)\s+(?:da\s+carris\s+)?(?:em|na|no|para|perto de)\s+(?P<term>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in stop_search_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {
                "name": "carris_get_stops",
                "args": {"query": _clean_query_fragment(match.group("term"))},
            }

    if stop_id_match and re.search(r"\b(arrivals|next arrivals|chegadas|pr[oó]ximas\s+chegadas)\b", query_lower):
        return {
            "name": "carris_get_arrivals",
            "args": {"stop_id": stop_id_match.group("stop_id")},
        }

    if stop_id_match and re.search(
        r"\b(next departures?|departures?|pr[oó]ximas?\s+partidas?)\b",
        query_lower,
    ):
        return {
            "name": "carris_get_next_departures",
            "args": {"stop_id": stop_id_match.group("stop_id")},
        }

    if route_code and re.search(
        r"\b(real-time|realtime|live|gps|location|locations|position|positions|where|onde|agora|now)\b",
        query_lower,
    ):
        return {
            "name": "carris_get_realtime_vehicles",
            "args": {"route_short_name": route_code},
        }

    if route_code and re.search(
        r"\b(frequency|headway|how often|interval|intervalo|frequ[eê]ncia|de quanto em quanto)\b",
        query_lower,
    ):
        return {
            "name": "carris_get_service_frequency",
            "args": {"route_short_name": route_code},
        }

    if route_code and re.search(
        r"\b(eta|estimated arrival|estimate the eta|when is the next|next\s+(?:bus|tram)|quando chega|chega a que horas)\b",
        query_lower,
    ):
        stop_match = re.search(
            r"\b(?:at|na|no|em)\s+(?P<stop>.+?)(?:[\?\!\.,;]|$)",
            query,
            flags=re.IGNORECASE,
        )
        if stop_match:
            return {
                "name": "carris_vehicle_eta",
                "args": {
                    "route_short_name": route_code,
                    "stop_name": _clean_query_fragment(stop_match.group("stop")),
                },
            }

    if route_code and re.search(
        r"\b(route information|route info|show route|show .*details|details|detalhes?|rota)\b",
        query_lower,
    ):
        return {"name": "carris_get_routes", "args": {"route_id": route_code}}

    if endpoints and re.search(
        r"\b(carris|tram|trams|el[eé]trico|eletrico|bus|buses|autocarro|autocarros)\b",
        query_lower,
    ):
        return {
            "name": "carris_find_routes_between",
            "args": {"origin": endpoints[0], "destination": endpoints[1], "search_radius_km": 0.8},
        }

    return None


def _query_has_status_intent(query: str) -> bool:
    """Returns whether the query is primarily asking about service status."""
    status_patterns = [
        r"\bis the metro working\b",
        r"\bcurrent status\b",
        r"\bmetro status\b",
        r"\btransport status\b",
        r"\bstatus overview\b",
        r"\bnetwork status\b",
        r"\bhow are the transports\b",
        r"\bpoint of situation\b",
        r"\bponto de situa[cç][aã]o\b",
        r"\bcomo est[aã]o os transportes\b",
        r"\best[aá] o metro a funcionar\b",
        r"\bestado do metro\b",
        r"\bestado dos transportes\b",
        r"\bponto de situa[cç][aã]o do metro\b",
        r"\bponto de situa[cç][aã]o dos transportes\b",
        r"\bperturba(?:cao|ções|coes|ção|caoes)s?\b.*\bmetro\b",
        r"\bperturba(?:cao|ções|coes|ção|caoes)s?\b.*\btransportes\b",
        r"\b(?:delay|delays|disruption|disruptions|problem|problems)\b.*\bmetro\b",
        r"\bmetro\b.*\b(?:delay|delays|disruption|disruptions|problem|problems)\b",
        r"\btransportes?\b.*\blisboa\b",
        r"\bmetro,?\s+autocarros?\s+e\s+comboios\b",
        r"\bare trains running\b",
        r"\bservice status\b",
    ]
    return any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in status_patterns)


def _query_is_aggregate_transport_status(query: str) -> bool:
    """Returns whether a status query asks for multiple transport families."""
    if _query_has_wait_departure_intent(query):
        return False

    query_lower = query.lower()
    status_hit = _query_has_status_intent(query) or bool(
        re.search(r"\b(?:status|situation|estado|situa[cç][aã]o|operational|operacional)\b", query_lower)
    )
    if not status_hit:
        return False

    mode_hits = 0
    mode_patterns = [
        r"\bmetro\b",
        r"\b(?:bus|buses|autocarro|autocarros)\b",
        r"\b(?:train|trains|comboio|comboios)\b",
        r"\b(?:tram|trams|el[eé]trico|eletrico)\b",
    ]
    for pattern in mode_patterns:
        if re.search(pattern, query_lower):
            mode_hits += 1

    broad_network = re.search(
        r"\b(?:transport|transports|transportes|network|rede|all transport|overview|resumo)\b",
        query_lower,
    )
    return mode_hits >= 2 or bool(broad_network and mode_hits >= 1)


def _query_has_wait_departure_intent(query: str) -> bool:
    """Returns whether the query asks for next departures, arrivals, wait times, or ETAs."""
    patterns = [
        r"\bnext\b",
        r"\bwait(?:ing)? times?\b",
        r"\bwhen is the next\b",
        r"\bdeparture(?:s)?\b",
        r"\barrival(?:s)?\b",
        r"\beta\b",
        r"\bpr[oó]xim[oa]s?\b",
        r"\btempos?\s+de\s+espera\b",
        r"\bchegada(?:s)?\b",
        r"\bpartida(?:s)?\b",
        r"\bquando passa\b",
        r"\bquanto tempo falta\b",
    ]
    return any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in patterns)


def _parse_metro_wait_blocks(wait_result: str) -> List[Dict[str, Any]]:
    """Parses structured wait-time blocks from Metro tool output."""
    blocks: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"Direction:\s*(?P<direction>[^\n]+)\n(?:\s*ℹ️\s*(?P<note>[^\n]+)\n)?\s*⏱️ Next train:\s*(?P<next>[^\n]+)\n\s*⏳ Following:\s*(?P<following>[^\n]+)",
        flags=re.IGNORECASE,
    )

    for match in pattern.finditer(wait_result or ""):
        following_parts = [
            part.strip() for part in match.group("following").split(",") if part.strip()
        ]
        blocks.append(
            {
                "direction": match.group("direction").strip(),
                "note": (match.group("note") or "").strip() or None,
                "times": [match.group("next").strip(), *following_parts],
            }
        )

    return blocks


def _parse_metro_line_wait_entries(wait_result: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Parses all-station Metro line wait snapshots from the Metro tool output."""
    entries: List[Dict[str, Any]] = []
    current_entry: Optional[Dict[str, Any]] = None
    updated_at: Optional[str] = None

    for raw_line in str(wait_result or "").splitlines():
        stripped = raw_line.strip()
        if not stripped or set(stripped) <= {"="}:
            continue

        updated_match = re.match(r"📍\s*Updated:\s*(?P<updated>.+)$", stripped, flags=re.IGNORECASE)
        if updated_match:
            updated_at = updated_match.group("updated").strip()
            current_entry = None
            continue

        station_match = re.match(r"📍\s*(?P<station>.+)$", stripped)
        if station_match:
            current_entry = {
                "station": _display_metro_line_station_name(station_match.group("station").strip()),
                "directions": [],
            }
            entries.append(current_entry)
            continue

        direction_match = re.match(r"→\s*(?P<direction>.+?):\s*(?P<wait>.+)$", stripped)
        if direction_match and current_entry is not None:
            current_entry["directions"].append(
                {
                    "direction": _display_metro_line_station_name(direction_match.group("direction").strip()),
                    "wait": direction_match.group("wait").strip(),
                }
            )

    return entries, updated_at


def _format_metro_line_wait_snapshot(
    *,
    line_id: str,
    wait_result: str,
    language: str,
    user_message: str = "",
) -> str:
    """Formats a Metro whole-line wait-time snapshot with localized labels."""
    line_name = _line_display_name(line_id, language)
    line_emoji = _metro_line_emoji(line_id)
    entries, updated_at = _parse_metro_line_wait_entries(wait_result)
    title_suffix = "Tempos de Espera" if language == "pt" else "Wait Times"
    title = f"### {line_emoji} **{line_name} - {title_suffix}**"

    if not entries:
        direct_answer = (
            f"não consegui confirmar agora os tempos de espera em tempo real para a **{line_name}**."
            if language == "pt"
            else f"I could not confirm real-time wait times for the **{line_name}** right now."
        )
        return "\n".join([title, "", f"✅ **{'Resposta direta' if language == 'pt' else 'Direct answer'}:** {direct_answer}"]).strip()

    try:
        from tools.metrolisboa_api import METRO_LINES

        station_ids = METRO_LINES.get(line_id, {}).get("stations", [])
    except Exception:
        station_ids = []
    terminal_text = ""
    if len(station_ids) >= 2:
        terminal_text = (
            f"{_display_metro_line_station_name(station_ids[0])} ↔ "
            f"{_display_metro_line_station_name(station_ids[-1])}"
        )

    snapshot_time = f" às **{updated_at[:5]}**" if language == "pt" and updated_at else ""
    snapshot_time_en = f" at **{updated_at[:5]}**" if language != "pt" and updated_at else ""
    compact_summary = _query_requests_compact_metro_wait_summary(user_message)

    if compact_summary:
        direct_answer = (
            f"este é um resumo curto dos tempos de espera em tempo real disponíveis para a **{line_name}**"
            f"{snapshot_time}, focado em terminais e estações de correspondência."
            if language == "pt"
            else f"this is a short summary of the real-time wait times available for the **{line_name}**"
            f"{snapshot_time_en}, focused on terminals and interchange stations."
        )
    else:
        direct_answer = (
            f"estes são os tempos de espera em tempo real disponíveis para a **{line_name}**"
            f"{snapshot_time}, em todas as estações reportadas."
            if language == "pt"
            else f"these are the real-time wait times available for the **{line_name}**"
            f"{snapshot_time_en}, across all reported stations."
        )

    direction_label = "Sentido" if language == "pt" else "Direction"
    line_summary_label = "Linha" if language == "pt" else "Line"

    lines = [
        title,
        "",
        f"✅ **{'Resposta direta' if language == 'pt' else 'Direct answer'}:** {direct_answer}",
        "",
        "---",
        "",
        f"{line_emoji} **{line_summary_label}:** {line_name}{f' — {terminal_text}' if terminal_text else ''}",
        "",
    ]

    entries_to_render = entries
    if compact_summary:
        try:
            from tools.metrolisboa_api import METRO_STATIONS
        except Exception:
            METRO_STATIONS = {}

        station_ids_normalized = [_normalize_token(station) for station in station_ids]
        terminal_keys = {
            station_ids_normalized[0],
            station_ids_normalized[-1],
        } if len(station_ids_normalized) >= 2 else set()

        def _is_key_station(entry: Dict[str, Any]) -> bool:
            station_key = _normalize_token(str(entry.get("station") or ""))
            if station_key in terminal_keys:
                return True
            lines_for_station = METRO_STATIONS.get(station_key) or []
            return len(lines_for_station) > 1

        key_entries = [entry for entry in entries if _is_key_station(entry)]
        if len(key_entries) < 4:
            midpoint = len(entries) // 2
            fallback_indexes = {0, midpoint, max(len(entries) - 1, 0)}
            for index in sorted(fallback_indexes):
                if 0 <= index < len(entries) and entries[index] not in key_entries:
                    key_entries.append(entries[index])
        entries_to_render = key_entries[:8] or entries[:6]

    for entry in entries_to_render:
        directions = entry.get("directions") or []
        if not directions:
            continue
        lines.append(f"**📍 {entry.get('station')}**")
        for direction in directions:
            wait_time = _localize_wait_times(str(direction.get("wait") or ""), language)
            lines.append(
                f"    - ➡️ **{direction_label} {direction.get('direction')}:** {wait_time}"
            )
        lines.append("")

    if compact_summary:
        lines.append(
            "💡 Para a lista completa, pede “todas as estações”."
            if language == "pt"
            else "💡 For the full list, ask for “all stations”."
        )

    return "\n".join(lines).strip()


def _format_all_metro_stations(*, language: str, user_message: str = "") -> str:
    """Formats Metro station inventories in the requested language."""
    from tools.metrolisboa_api import METRO_LINES

    requested_line_id = _extract_metro_line_id(user_message)
    normalized_message = _normalize_token(user_message)
    orientation_intent = bool(
        re.search(
            r"\b(?:orient|orientation|understand\s+(?:the\s+)?(?:network|metro)|"
            r"main\s+(?:stations|interchanges|hubs)|key\s+(?:stations|interchanges|hubs)|"
            r"interchanges?|hubs?|transbordos?|correspond[eê]ncias?|esta[cç][oõ]es\s+principais|"
            r"pontos?\s+principais)\b",
            normalized_message,
        )
    )
    explicit_inventory_intent = bool(
        re.search(
            r"\b(?:all|every|todas?|todos?|list(?:ar)?|show|mostrar)\b.*\b(?:stations|esta[cç][oõ]es|estacoes)\b",
            normalized_message,
        )
    )
    line_ids = [requested_line_id] if requested_line_id else ["amarela", "azul", "verde", "vermelha"]
    line_ids = [line_id for line_id in line_ids if line_id in METRO_LINES]

    interchange_prefixes = {
        "campo grande": "🟡🟢",
        "alameda": "🟢🔴",
        "saldanha": "🔴🟡",
        "marques de pombal": "🟡🔵",
        "baixa chiado": "🔵🟢",
        "baixa-chiado": "🔵🟢",
        "sao sebastiao": "🔵🔴",
    }
    rail_prefixes = {
        "santa apolonia": "🚆",
        "cais do sodre": "🚆",
        "rossio": "🚆",
        "areeiro": "🚆",
        "oriente": "🚆",
    }

    def station_label(station_name: str) -> str:
        display_name = _display_metro_line_station_name(station_name)
        normalized = _normalize_token(station_name)
        prefix = interchange_prefixes.get(normalized) or rail_prefixes.get(normalized)
        return f"{prefix} {display_name}" if prefix else display_name

    if orientation_intent and not requested_line_id and not explicit_inventory_intent:
        if language == "pt":
            return "\n".join(
                [
                    "### 🚇 **Orientação rápida no Metro de Lisboa**",
                    "",
                    "✅ **Resposta direta:** usa estes nós como pontos de referência; não precisas de decorar todas as estações.",
                    "",
                    "---",
                    "",
                    "- 🔴 **Aeroporto:** entrada direta na Linha Vermelha.",
                    "- 🟢🔴 **Alameda:** ligação Vermelha ↔ Verde para Baixa/Cais do Sodré ou Areeiro/Roma.",
                    "- 🔴🟡 **Saldanha:** ligação Vermelha ↔ Amarela para Entrecampos, Campo Grande, Rato ou Odivelas.",
                    "- 🔵🔴 **São Sebastião:** ligação Vermelha ↔ Azul para Marquês, Baixa-Chiado, Terreiro do Paço ou Reboleira.",
                    "- 🟡🔵 **Marquês de Pombal:** nó central para alternar entre Amarela e Azul.",
                    "- 🔵🟢 **Baixa-Chiado:** nó central para Chiado, Baixa e ligação Azul ↔ Verde.",
                    "- 🟡🟢 **Campo Grande:** ligação Amarela ↔ Verde e referência útil para norte da cidade.",
                    "- 🚆 **Oriente / Cais do Sodré / Santa Apolónia:** interfaces úteis quando também precisas de comboios.",
                    "",
                    "💡 **Dica rápida:** a partir do Aeroporto, a Linha Vermelha liga-te aos principais transbordos: Alameda, Saldanha e São Sebastião.",
                ]
            ).strip()
        return "\n".join(
            [
                "### 🚇 **Quick Lisbon Metro Orientation**",
                "",
                "✅ **Direct answer:** use these hubs as reference points; you do not need the full station list to understand the network.",
                "",
                "---",
                "",
                "- 🔴 **Aeroporto:** direct entry point on the Red line.",
                "- 🟢🔴 **Alameda:** Red ↔ Green interchange for Baixa/Cais do Sodré or Areeiro/Roma.",
                "- 🔴🟡 **Saldanha:** Red ↔ Yellow interchange for Entrecampos, Campo Grande, Rato, or Odivelas.",
                "- 🔵🔴 **São Sebastião:** Red ↔ Blue interchange for Marquês, Baixa-Chiado, Terreiro do Paço, or Reboleira.",
                "- 🟡🔵 **Marquês de Pombal:** central Yellow ↔ Blue interchange.",
                "- 🔵🟢 **Baixa-Chiado:** central hub for Chiado, Baixa, and Blue ↔ Green transfers.",
                "- 🟡🟢 **Campo Grande:** Yellow ↔ Green interchange and useful northern reference point.",
                "- 🚆 **Oriente / Cais do Sodré / Santa Apolónia:** useful interfaces when you also need rail services.",
                "",
                "💡 **Quick tip:** from Aeroporto, the Red line takes you to the key interchanges: Alameda, Saldanha, and São Sebastião.",
            ]
        ).strip()

    if requested_line_id:
        line_name = _line_display_name(requested_line_id, language)
        line_emoji = _metro_line_emoji(requested_line_id)
        title = (
            f"### {line_emoji} **Estações da {line_name}**"
            if language == "pt"
            else f"### {line_emoji} **{line_name} Stations**"
        )
        direct_answer = (
            f"aqui estão as estações da **{line_name}**, mantendo os principais transbordos assinalados."
            if language == "pt"
            else f"here are the **{line_name}** stations, with key interchanges marked."
        )
    else:
        title = "### 🚇 **Estações do Metro de Lisboa**" if language == "pt" else "### 🚇 **Lisbon Metro Stations**"
        direct_answer = (
            "aqui estão as estações do Metro de Lisboa organizadas por linha."
            if language == "pt"
            else "here are Lisbon Metro stations organized by line."
        )

    lines = [
        title,
        "",
        f"✅ **{'Resposta direta' if language == 'pt' else 'Direct answer'}:** {direct_answer}",
        "",
        "---",
        "",
    ]

    for line_id in line_ids:
        line_info = METRO_LINES.get(line_id, {})
        stations = list(line_info.get("stations", []))
        if line_id == "azul":
            stations.reverse()
        if not stations:
            continue
        line_name = _line_display_name(line_id, language)
        line_emoji = _metro_line_emoji(line_id)
        terminal_text = f"{_display_metro_line_station_name(stations[0])} ↔ {_display_metro_line_station_name(stations[-1])}"
        station_text = " · ".join(station_label(station) for station in stations)
        lines.extend(
            [
                f"{line_emoji} **{line_name}:** {terminal_text}",
                f"    - {station_text}",
                "",
            ]
        )

    if language == "pt":
        lines.extend(
            [
                "ℹ️ **Legenda:**",
                "    - 🟡🟢, 🟢🔴, 🔴🟡, 🟡🔵, 🔵🟢 e 🔵🔴 assinalam estações de correspondência entre linhas.",
                "    - 🚆 assinala estações servidas por serviços ferroviários ou ligadas diretamente a eles.",
            ]
        )
    else:
        lines.extend(
            [
                "ℹ️ **Legend:**",
                "    - 🟡🟢, 🟢🔴, 🔴🟡, 🟡🔵, 🔵🟢 and 🔵🔴 mark line-interchange stations.",
                "    - 🚆 marks stations served by rail or directly connected to rail services.",
            ]
        )

    return "\n".join(lines).strip()


def _extract_wait_block_for_direction(wait_result: str, target_direction: str) -> Optional[Dict[str, Any]]:
    """Extracts the full wait-time block for a requested metro direction."""
    target_norm = _normalize_token(target_direction)
    for block in _parse_metro_wait_blocks(wait_result):
        if _normalize_token(block["direction"]) == target_norm:
            return block
    return None


def _localize_platform_note(note: Optional[str], language: str) -> Optional[str]:
    """Localizes metro platform notes when possible."""
    if not note:
        return None
    if language != "pt":
        return note

    match = re.match(
        r"Platform indicator currently shows\s+(.+?)\.?$",
        note,
        flags=re.IGNORECASE,
    )
    if match:
        return f"O indicador de plataforma mostra de momento {match.group(1).strip()}."
    return note


def _build_metro_wait_lines(targets: List[Tuple[str, str]], language: str) -> List[str]:
    """Builds rich metro wait lines while preserving platform notes and all reported times."""
    from tools.metrolisboa_api import get_metro_wait_time

    station_label = "Estação" if language == "pt" else "Station"
    direction_label = "Direção" if language == "pt" else "Direction"
    next_label = "⏱️ Próximo Metro em:" if language == "pt" else "⏱️ Next Metro in:"

    realtime_lines: List[str] = []
    official_api_temporarily_unavailable = False
    for station, direction in targets[:2]:
        try:
            wait_result = str(
                get_metro_wait_time.invoke(
                    {"station": station, "direction": direction}
                )
            )
        except Exception:
            wait_result = ""

        wait_result_normalized = _normalize_token(wait_result)
        if wait_result.strip().startswith("❌") and any(
            marker in wait_result_normalized
            for marker in [
                "temporarily unavailable",
                "temporariamente indisponivel",
                "not responding right now",
                "official metro",
                "oficial metro",
                "fallback endpoint still provides line status",
            ]
        ):
            official_api_temporarily_unavailable = True
            continue

        block = _extract_wait_block_for_direction(wait_result, direction)
        if not block:
            continue

        localized_times = [
            _localize_wait_times(time_text, language) for time_text in block.get("times", [])
        ]
        station_display = _display_metro_line_station_name(station)
        direction_display = _display_metro_line_station_name(direction)
        realtime_lines.append(
            f"- **{station_label} {station_display}:** {direction_label} {direction_display} — **{next_label}** {' | '.join(localized_times)}"
        )
        note = _localize_platform_note(block.get("note"), language)
        if note:
            realtime_lines.append(f"    - ℹ️ {note}")

    if not realtime_lines:
        if official_api_temporarily_unavailable:
            if language == "pt":
                realtime_lines.extend(
                    [
                        "- Dados oficiais do Metro em tempo real estão temporariamente indisponíveis.",
                        "  ℹ️ O estado das linhas continua disponível, mas os tempos de espera não puderam ser confirmados agora.",
                    ]
                )
            else:
                realtime_lines.extend(
                    [
                        "- Official Metro real-time data is temporarily unavailable.",
                        "  ℹ️ Line status is still available, but live wait times could not be confirmed right now.",
                    ]
                )
        else:
            realtime_lines.append(
                "- Sem dados em tempo real" if language == "pt" else "- No real-time data available"
            )

    return realtime_lines


def _extract_route_endpoints(
    user_message: str,
    *,
    collapse_generic_service_area: bool = True,
) -> Optional[Tuple[str, str]]:
    """Extracts route endpoints from common PT/EN route phrasings."""
    if _query_requests_metro_line_wait_times(user_message):
        return None

    route_text = _expand_route_address_abbreviations(user_message)

    shorthand_pair = _extract_metro_station_pair_from_shorthand(route_text)
    if shorthand_pair:
        return shorthand_pair

    if _looks_like_itinerary_request_without_explicit_route(route_text):
        return None

    patterns = [
        r"(?P<origin>.+?)\s*(?:->|→|=>)\s*(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\b(?:i\s+want\s+to\s+(?:go|get|travel)|we\s+want\s+to\s+(?:go|get|travel)|"
        r"how\s+(?:do|can)\s+i\s+(?:get|go|travel)|how\s+to\s+(?:get|go|travel))?\s*"
        r"from\s+(?P<origin>.+?)\s+to\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\b(?:quero|preciso|tenho)\s+(?:de\s+)?ir\s+(?:desde|a\s+partir\s+(?:dos|das|do|da|de))\s+(?P<origin>.+?)\s+(?:at[eé]|para|ao|a|à)\s+(?P<destination>.+?)(?:\.\s+(?:diz[- ]me|diga[- ]me|tell\s+me|show\s+me)\b|[\?\!;]|$)",
        r"\b(?:quais\s+os\s+)?(?:pr[oó]xim(?:os|as)|next)\s+(?:autocarros?|buses?|el[eé]tricos?|trams?).*?\b(?:at|em|na|no)\s+(?P<origin>.+?)\s+(?:para\s+(?:seguir\s+)?(?:para\s+)?|to\s+)(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\b(?:plan|planeia|planeie|organiza|organize)\b.*?\b(?:em|in)\s+(?P<destination>[^,\?\!\.]+?)\s+"
        r"(?:a\s+partir\s+(?:dos|das|do|da|de)|desde|from|starting\s+from|start\s+from|"
        r"starting\s+at|come[cç]ando\s+(?:em|no|na|nos|nas|do|da|dos|das|de))\s+"
        r"(?P<origin>[^,\?\!\.]+?)(?:\s*,|\s+com\b|\s+with\b|\s+e\b|\s+and\b|[\?\!\.]|$)",
        r"\b(?:plan|planeia|planeie|organiza|organize)\b.*?\b(?:em|in)\s+(?P<destination>[^,\?\!\.]+),\s*(?:diz[- ]me|diga[- ]me|tell me|show me)\s+como\s+l[aá]\s+chegar\s+a\s+partir\s+(?:dos|das|do|da|de)\s+(?P<origin>.+?)(?:\s+e\b|[\?\!\.,;]|$)",
        r"\b(?:plan|planeia|planeie|organiza|organize)\b.*?\b(?:em|in)\s+(?P<destination>[^,\?\!\.]+),\s*(?:tell me|show me)\s+how\s+to\s+get\s+there\s+from\s+(?P<origin>.+?)(?:\s+and\b|[\?\!\.,;]|$)",
        r"\b(?:como\s+(?:é\s+que\s+)?(?:fa[cç]o\s+para\s+)?(?:posso\s+)?ir|como\s+(?:é\s+que\s+)?vou|como\s+chego)\s+(?:dos|das|do|da|de)\s+(?P<origin>.+?)\s+(?:para|ao|a|à|até)\s+(?P<destination>.+?)(?:\s*\?\s*(?:d[aá]\s+para|posso|can\s+i)\b|[\?\!\.,;]|$)",
        r"\b(?:como\s+(?:é\s+que\s+)?(?:posso\s+)?ir|como\s+(?:é\s+que\s+)?vou|como\s+chego)\s+(?:dos|das|do|da|de)\s+(?P<origin>.+?)\s+(?:para|ao|a|à|até)\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\b(?:quero\s+)?(?:sair|partir)\s+(?:dos|das|do|da|de)\s+(?P<origin>.+?)\s+(?:e\s+)?(?:ir|seguir|chegar)\s+(?:para|ao|a|à|até)\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\b(?:quero|preciso|tenho)\s+(?:de\s+)?ir\s+(?:dos|das|do|da|de)\s+(?P<origin>.+?)\s+(?:para|ao|a|à|até)\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\b(?:quero|preciso|tenho)\s+(?:de\s+)?ir\s+(?:para|ao|a|à)\s+(?P<destination>.+?)\s+a\s+partir\s+(?:dos|das|do|da|de)\s+(?P<origin>.+?)(?:[\?\!\.,;]|$)",
        r"\b(?:a\s+partir\s+(?:dos|das|do|da|de)|desde)\s+(?P<origin>.+?)\s+(?:at[eé]|para|ao|a|à)\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\b(?:tou|estou)\s+n(?:o|a)\s+(?P<origin>.+?)\s+(?:e\s+)?(?:preciso|quero|tenho)\s+(?:de\s+)?ir\s+(?:para|ao|a|à)\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\b(?:need|want)\s+to\s+(?:go|get)\s+to\s+(?P<destination>.+?)\s+from\s+(?P<origin>.+?)(?:[\?\!\.,;]|$)",
        r"\b(?:i'?m|i am)\s+(?:at|in)\s+(?P<origin>.+?)\s+and\s+(?:i\s+)?(?:need|want)\s+to\s+(?:go|get)\s+to\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bde\s+metro\s+de\s+(?P<origin>.+?)\s+at[eé]\s+(?:a|ao|à)?\s*(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bde\s+metro\s+de\s+(?P<origin>.+?)\s+para\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bdos\s+(?P<origin>.+?)\s+a\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bdas\s+(?P<origin>.+?)\s+a\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bdo\s+(?P<origin>.+?)\s+a\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bda\s+(?P<origin>.+?)\s+a\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bde\s+(?P<origin>.+?)\s+at[eé]\s+(?:a|ao|à)?\s*(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bde\s+(?P<origin>.+?)\s+para\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bde\s+(?P<origin>.+?)\s+a\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bdos\s+(?P<origin>.+?)\s+ao\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bdos\s+(?P<origin>.+?)\s+para\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bdas\s+(?P<origin>.+?)\s+à\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bdas\s+(?P<origin>.+?)\s+para\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bdo\s+(?P<origin>.+?)\s+ao\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bdo\s+(?P<origin>.+?)\s+para\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bda\s+(?P<origin>.+?)\s+à\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bda\s+(?P<origin>.+?)\s+para\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bde\s+(?P<origin>.+?)\s+ao\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bde\s+(?P<origin>.+?)\s+à\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bfrom\s+(?P<origin>.+?)\s+to\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bbetween\s+(?P<origin>.+?)\s+and\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bentre\s+(?P<origin>.+?)\s+e\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
    ]

    def _valid_endpoint(value: str) -> bool:
        """Reject parser artefacts such as the ``e/`` in ``metro e/ou autocarro``."""
        normalized = _normalize_token(value)
        if len(normalized) < 3:
            return False
        if normalized in {"e", "ou", "and", "or", "e/ou", "and/or"}:
            return False
        if re.fullmatch(r"[a-z]/?", normalized):
            return False
        planner_fragment_terms = (
            "dia", "dias", "day", "days", "roteiro", "itinerario", "itinerary",
            "plano", "plan", "casal", "senior", "ritmo", "orcamento", "budget",
            "gastronomia", "museus", "monumentos", "chuva", "chover", "rain", "rainy",
            "caminhada", "caminhar", "walking", "walk", "andar", "transbordo",
            "transbordos", "transferencia", "transferencias", "escadas", "bagagem",
            "luggage", "mala", "malas", "idosos", "elderly", "criancas", "children",
            "kids", "preferencias", "preferences",
        )
        if any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in planner_fragment_terms):
            return False
        if len(normalized.split()) > 8 and not re.search(
            r"\b(?:rua|avenida|av|praca|praça|largo|station|estacao|estação|metro|aeroporto|airport)\b",
            normalized,
        ):
            return False
        return True

    for pattern in patterns:
        match = re.search(pattern, route_text, flags=re.IGNORECASE)
        if match:
            origin = _clean_query_fragment(match.group("origin"))
            destination = _clean_query_fragment(match.group("destination"))
            if collapse_generic_service_area:
                destination = _generic_service_area_endpoint(destination) or destination
            if origin and destination and _valid_endpoint(origin) and _valid_endpoint(destination):
                return origin, destination

    return None


def _looks_like_itinerary_request_without_explicit_route(user_message: str) -> bool:
    """Return whether broad itinerary prose should not be parsed as a route.

    Planning prompts often contain grammatical fragments like ``roteiro de 2
    dias ... para um casal``. The generic ``de X para Y`` route parser must not
    treat those as origin/destination pairs unless the user also asks an
    explicit mobility question.
    """
    normalized = _normalize_token(user_message)
    if not normalized:
        return False

    planning_signal = bool(
        re.search(
            r"\b(?:roteiro|itinerario|itinerary|plano|planear|planeia|organiza|organize|cria|criar|tour)\b",
            normalized,
        )
        or re.search(r"\b\d+\s*(?:dias?|days?)\b", normalized)
    )
    if not planning_signal:
        return False

    explicit_route_signal = bool(
        re.search(
            r"\b(?:como\s+(?:vou|chego)|how\s+(?:do\s+i\s+)?(?:get|go)|"
            r"ir\s+(?:de|do|da|dos|das)|vou\s+(?:de|do|da|dos|das)|"
            r"quero\s+(?:ir|chegar)|preciso\s+(?:de\s+)?ir|tenho\s+(?:de\s+)?ir|"
            r"route\s+from|from\s+.+\s+to|between\s+.+\s+and|entre\s+.+\s+e|"
            r"a\s+partir\s+(?:de|do|da|dos|das)|desde|de\s+metro\s+de|de\s+autocarro\s+de|de\s+comboio\s+de)\b",
            normalized,
        )
        or re.search(r"(?:->|→|=>)", normalized)
    )
    return not explicit_route_signal


def _extract_metro_station_pair_from_shorthand(user_message: str) -> Optional[Tuple[str, str]]:
    """Extract a metro origin/destination pair from shorthand prompts like 'ML azul baixa chiado rato'."""
    normalized_query = _normalize_token(user_message)
    if not re.match(r"^(?:ml|metro)\b", normalized_query):
        return None

    shorthand = re.sub(r"^(?:ml|metro)\s+", "", normalized_query)
    shorthand = re.sub(r"^(?:azul|blue|verde|green|amarela|yellow|vermelha|red)\s+", "", shorthand)
    station_aliases = _get_metro_station_name_map()
    hits: List[Tuple[int, int, str]] = []
    for alias, canonical in station_aliases.items():
        index = shorthand.find(alias)
        if index < 0:
            continue
        hits.append((index, len(alias), canonical))

    if len(hits) < 2:
        return None

    hits.sort(key=lambda item: (item[0], -item[1], item[2]))
    selected: List[Tuple[int, int, str]] = []
    occupied: List[Tuple[int, int]] = []
    for index, length, canonical in hits:
        span_end = index + length
        if any(index < used_end and span_end > used_start for used_start, used_end in occupied):
            continue
        if selected and _normalize_token(selected[-1][2]) == _normalize_token(canonical):
            continue
        selected.append((index, length, canonical))
        occupied.append((index, span_end))

    if len(selected) < 2:
        return None

    origin = selected[0][2]
    destination = selected[-1][2]
    if _normalize_token(origin) == _normalize_token(destination):
        return None
    return origin, destination


def _extract_destination_only_target(user_message: str) -> Optional[str]:
    """Extract a destination when the user asks for nearby transport options without an origin."""
    if _extract_route_endpoints(user_message):
        return None
    if _query_has_wait_departure_intent(user_message):
        return None
    if re.search(
        r"\b(?:by|de)\s+(?:metro|bus|buses|autocarro|autocarros|train|comboio|comboios|tram|trams|el[eé]trico|el[eé]tricos)\b",
        user_message,
        flags=re.IGNORECASE,
    ):
        return None

    mode_preferences = _parse_route_mode_preferences(user_message)
    if any(mode_preferences.values()):
        return None

    query = user_message.strip()
    patterns = [
        r"\b(?:transport to|how do i get to|how to get to)\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\b(?:como vou|como chego|quero ir)\s+(?:para|ao|à|a)\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\btransportes?\s+dispon[ií]veis?\s+(?:em|para)?\s*(?P<destination>.+?)(?:[\?\!\.,;]|$)",
        r"\bavailable\s+transport\s+(?:in|near|for)\s+(?P<destination>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return _clean_query_fragment(match.group("destination"))

    return None


def _build_destination_only_transport_overview_response(
    user_message: str,
    context: str,
) -> Optional[str]:
    """Build a nearby-options overview for single-destination transport questions."""
    destination = _extract_destination_only_target(user_message)
    if not destination:
        return None

    try:
        from tools.transport_api import find_nearest_stops_for_place
    except ImportError:
        return None

    overview = find_nearest_stops_for_place(destination)
    if not overview:
        return None

    language = _infer_language(user_message, context)
    display_name = str(overview.get("display_name") or destination).strip()
    metro_name = str(overview.get("metro") or "").strip()
    metro_line = str(overview.get("metro_line") or "").strip()
    metro_walk_minutes = overview.get("metro_walk_minutes")
    train_station = str(overview.get("train_station") or "").strip()
    train_walk_minutes = overview.get("train_walk_minutes")
    carris_stops = overview.get("carris_stops") or []
    if not metro_name and not train_station and not carris_stops:
        return None

    title = (
        f"🧭 **Transportes disponíveis perto de {display_name}**"
        if language == "pt"
        else f"🧭 **Transport options near {display_name}**"
    )
    lines = [title]
    source_links: List[str] = []

    if metro_name:
        metro_bits = [f"🚇 **Metro mais próximo:** **{metro_name.title()}**" if language == "pt" else f"🚇 **Nearest Metro:** **{metro_name.title()}**"]
        if metro_line:
            metro_bits.append(_format_metro_line_set(metro_line, language))
        if metro_walk_minutes:
            walk_suffix = (
                f"(~{metro_walk_minutes} min a pé)"
                if language == "pt"
                else f"(~{metro_walk_minutes} min walk)"
            )
            metro_bits.append(walk_suffix)
        lines.extend(["", " · ".join(metro_bits)])
        source_links.append("[*Metro de Lisboa*](https://www.metrolisboa.pt)")

    if train_station:
        train_line = (
            f"🚆 **Estação CP mais próxima:** **{train_station}**"
            if language == "pt"
            else f"🚆 **Nearest CP station:** **{train_station}**"
        )
        if train_walk_minutes:
            train_line += (
                f" (~{train_walk_minutes} min a pé)"
                if language == "pt"
                else f" (~{train_walk_minutes} min walk)"
            )
        lines.extend(["", train_line])
        source_links.append("[*CP*](https://www.cp.pt)")

    if carris_stops:
        lines.extend([
            "",
            "🚌 **Paragens Carris próximas:**" if language == "pt" else "🚌 **Nearby Carris stops:**",
        ])
        for stop in carris_stops:
            distance_km = float(stop.get("distance_km") or 0.0)
            distance_label = (
                f"{round(distance_km * 1000):.0f} m"
                if distance_km < 1
                else f"{distance_km:.1f} km"
            )
            lines.append(f"- **{stop['stop_name']}** · {distance_label}")
        source_links.append("[*Carris*](https://www.carris.pt)")

    lines.extend([
        "",
        "💡 **Dica rápida:** Se me disseres a origem, calculo o percurso completo a partir deste destino."
        if language == "pt"
        else "💡 **Quick Tip:** If you tell me your origin, I can turn this into a full route from that destination.",
        "",
        (
            f"📌 **Fonte:** {' | '.join(dict.fromkeys(source_links))} | **Atualizado:** {datetime.now().strftime('%H:%M')}"
            if language == "pt"
            else f"📌 **Source:** {' | '.join(dict.fromkeys(source_links))} | **Updated:** {datetime.now().strftime('%H:%M')}"
        ),
    ])
    return "\n".join(lines).strip()


def _is_preformatted_metro_route_response(response: str) -> bool:
    """Return whether a deterministic Metro route is already in its final structured layout."""
    text = str(response or "")
    if not text:
        return False

    has_route_header = any(
        marker in text
        for marker in ["🗺️ **O seu Trajeto de Metro:**", "🗺️ **Your Metro Route:**"]
    )
    has_wait_section = any(
        marker in text
        for marker in ["🗓️ **Próximos Metros", "🗓️ **Next Metros"]
    )
    return (
        (text.startswith("🚇 **") or text.startswith("### 🚇 **"))
        and "⏳ **" in text
        and has_route_header
        and has_wait_section
    )


def _parse_route_mode_preferences(user_message: str) -> Dict[str, bool]:
    """Parses explicit route-mode constraints such as bus-only or without-bus requests."""
    normalized = _normalize_token(user_message)

    bus_only_phrases = [
        "bus only",
        "only bus",
        "only buses",
        "by bus only",
        "only by bus",
        "so autocarro",
        "so autocarros",
        "só autocarro",
        "só autocarros",
        "so de autocarro",
        "só de autocarro",
        "so de autocarros",
        "só de autocarros",
        "apenas autocarro",
        "apenas autocarros",
        "apenas de autocarro",
        "apenas de autocarros",
    ]
    tram_only_phrases = [
        "tram only",
        "only tram",
        "only trams",
        "by tram only",
        "only by tram",
        "so eletrico",
        "so eletricos",
        "so electrico",
        "só eletrico",
        "só elétrico",
        "só eletricos",
        "só elétricos",
        "so de eletrico",
        "só de elétrico",
        "so de eletricos",
        "só de elétricos",
        "apenas eletrico",
        "apenas elétrico",
        "apenas eletricos",
        "apenas elétricos",
        "apenas de eletrico",
        "apenas de elétrico",
    ]
    metro_only_phrases = [
        "metro only",
        "only metro",
        "only by metro",
        "by metro only",
        "so metro",
        "só metro",
        "so de metro",
        "só de metro",
        "apenas metro",
        "apenas de metro",
    ]
    exclude_bus_phrases = [
        "without bus",
        "without buses",
        "without taking a bus",
        "without taking buses",
        "without using a bus",
        "without using buses",
        "no bus",
        "no buses",
        "not by bus",
        "dont want bus",
        "do not want bus",
        "sem autocarro",
        "sem autocarros",
        "nao quero autocarro",
        "não quero autocarro",
        "nao quero autocarros",
        "não quero autocarros",
    ]
    exclude_tram_phrases = [
        "without tram",
        "without trams",
        "without taking a tram",
        "without taking trams",
        "no tram",
        "no trams",
        "not by tram",
        "sem eletrico",
        "sem elétrico",
        "sem eletricos",
        "sem elétricos",
        "sem electrico",
        "nao quero eletrico",
        "não quero elétrico",
        "nao quero eletricos",
        "não quero elétricos",
    ]
    exclude_metro_phrases = [
        "without metro",
        "without taking the metro",
        "without taking metro",
        "without using the metro",
        "avoiding metro",
        "no metro",
        "not by metro",
        "avoid metro",
        "evitar metro",
        "evitando metro",
        "evito metro",
        "prefiro evitar metro",
        "sem metro",
        "sem usar metro",
        "nao quero metro",
        "não quero metro",
        "nao usar metro",
        "não usar metro",
    ]

    bus_only = any(phrase in normalized for phrase in bus_only_phrases)
    tram_only = any(phrase in normalized for phrase in tram_only_phrases)
    metro_only = any(phrase in normalized for phrase in metro_only_phrases)
    exclude_bus = any(phrase in normalized for phrase in exclude_bus_phrases)
    exclude_tram = any(phrase in normalized for phrase in exclude_tram_phrases)
    exclude_metro = any(phrase in normalized for phrase in exclude_metro_phrases)

    exclusion_prefix = r"(?:sem|evitar|evitando|evito|dispensar|prefiro\s+evitar|n[aã]o\s+quero|without|avoid|avoiding|no)"
    optional_use = r"(?:\s+(?:usar|apanhar|tomar|using|taking|take|catch))?"
    exclude_bus = exclude_bus or bool(
        re.search(
            rf"\b{exclusion_prefix}{optional_use}\s+(?:o\s+|a\s+|the\s+)?(?:bus|buses|autocarro|autocarros)\b",
            normalized,
        )
    )
    exclude_tram = exclude_tram or bool(
        re.search(
            rf"\b{exclusion_prefix}{optional_use}\s+(?:o\s+|a\s+|the\s+)?(?:tram|trams|eletrico|eletricos|electrico|electricos)\b",
            normalized,
        )
    )
    exclude_metro = exclude_metro or bool(
        re.search(
            rf"\b{exclusion_prefix}{optional_use}\s+(?:o\s+|a\s+|the\s+)?metro\b",
            normalized,
        )
    )

    bus_only = bus_only or bool(
        re.search(
            r"\b(?:only|just|apenas|s[oó])\b(?:\s+de)?\s+(?:bus|buses|autocarro|autocarros)\b",
            normalized,
        )
        or re.search(
            r"\b(?:outros?|outras?|mais|another|other|more)\s+(?:bus|buses|autocarros?|linhas?\s+de\s+autocarro)\b",
            normalized,
        )
    )
    tram_only = tram_only or bool(
        re.search(
            r"\b(?:only|just|apenas|s[oó])\b(?:\s+de)?\s+(?:tram|trams|eletrico|eletricos|electrico|electricos)\b",
            normalized,
        )
        or re.search(
            r"\b(?:outros?|outras?|mais|another|other|more)\s+(?:tram|trams|el[eé]tricos?|linhas?\s+de\s+el[eé]trico)\b",
            normalized,
        )
    )
    metro_only = metro_only or bool(
        re.search(
            r"\b(?:only|just|apenas|s[oó])\b(?:\s+de)?\s+metro\b",
            normalized,
        )
    )
    metro_only = metro_only or bool(
        re.match(r"^(?:ml|metro)\s+(?:azul|blue|verde|green|amarela|yellow|vermelha|red)\b", normalized)
    )

    route_mode_pattern = (
        r"(?:metro|bus|buses|autocarro|autocarros|tram|trams|eletrico|eletricos|"
        r"comboio|comboios|train|trains)"
    )
    alternative_mode_request = bool(
        re.search(
            rf"\b{route_mode_pattern}\b.{{0,40}}\b(?:e/ou|e|ou|or|and|vs|versus)\b.{{0,40}}\b{route_mode_pattern}\b",
            normalized,
        )
        or re.search(
            r"\b(?:alternativas?|op[cç][oõ]es|outra\s+op[cç][aã]o|outros?\s+meios?\s+de\s+transporte|"
            r"outros?\s+transportes?|meios?\s+de\s+transporte|compara(?:r)?\s+(?:os\s+)?meios|"
            r"alternatives?|options?|another\s+option|other\s+(?:transport|transit)\s+modes?|"
            r"compare\s+(?:the\s+)?(?:transport|transit)\s+modes?)\b",
            normalized,
        )
    )
    if not alternative_mode_request:
        bus_only = bus_only or bool(
            re.search(
                r"\b(?:que|qual|which|what)\s+(?:bus|buses|autocarro|autocarros)\s+"
                r"(?:(?:devo|deveria|posso|should(?:\s+i)?|can(?:\s+i)?|do\s+i)\s+)?"
                r"(?:apanhar|apanho|tomar|usar|take|catch|use)\b|\b"
                r"(?:apanhar|apanho|tomar|usar|take|catch|use)\s+"
                r"(?:o\s+|a\s+|the\s+)?(?:bus|buses|autocarro|autocarros)\b",
                normalized,
            )
        )
        bus_only = bus_only or bool(
            re.search(
                r"\b(?:de|by|via|using)\s+(?:o\s+|a\s+|the\s+)?(?:bus|buses|autocarro|autocarros)\b",
                normalized,
            )
        )
        tram_only = tram_only or bool(
            re.search(
                r"\b(?:de|by|via|using)\s+(?:o\s+|a\s+|the\s+)?(?:tram|trams|eletrico|eletricos|electrico|electricos)\b",
                normalized,
            )
        )
        metro_only = metro_only or bool(
            re.search(
                r"\b(?:de|by|via|using)\s+(?:o\s+|a\s+|the\s+)?metro\b|\bmetro\s+(?:entre|between|from|de)\b",
                normalized,
            )
        )

    if exclude_tram and re.search(r"\b(bus|buses|autocarro|autocarros)\b", normalized):
        bus_only = True
    if exclude_bus and re.search(r"\b(tram|trams|eletrico|eletricos|electrico|electricos)\b", normalized):
        tram_only = True
    if exclude_metro and re.search(r"\b(bus|buses|autocarro|autocarros)\b", normalized):
        bus_only = True
    if exclude_metro and re.search(r"\b(tram|trams|eletrico|eletricos|electrico|electricos)\b", normalized):
        tram_only = True

    return {
        "bus_only": bus_only,
        "tram_only": tram_only,
        "metro_only": metro_only,
        "exclude_bus": exclude_bus,
        "exclude_tram": exclude_tram,
        "exclude_metro": exclude_metro,
        "alternative_mode_request": alternative_mode_request,
    }


def _query_has_route_mode_constraints(user_message: str) -> bool:
    """Returns whether the user explicitly constrained the allowed transport modes for a route."""
    preferences = _parse_route_mode_preferences(user_message)
    preferences = {
        key: value
        for key, value in preferences.items()
        if key != "alternative_mode_request"
    }
    return any(preferences.values())


def _query_has_route_quality_preferences(user_message: str) -> bool:
    """Return whether a route asks for qualitative optimization beyond modes.

    These requests should still use deterministic transport tools for evidence,
    but should avoid the shortest static fast path so the LLM can synthesize
    trade-offs and explicit limitations from the gathered route data.
    """
    normalized = _normalize_token(user_message)
    if not normalized:
        return False
    return bool(
        re.search(
            r"\b(?:menos\s+caminhada|pouca\s+caminhada|caminhar\s+pouco|least\s+walking|"
            r"less\s+walking|minimal\s+walking|menos\s+transbordos?|menos\s+transferencias?|"
            r"menos\s+transferências?|fewest\s+transfers?|least\s+transfers?|sem\s+escadas|"
            r"step[-\s]?free|acessivel|acessível|accessible|cadeira\s+de\s+rodas|wheelchair|"
            r"mobilidade\s+reduzida|reduced\s+mobility|com\s+chuva|se\s+chover|rain|rainy|"
            r"bagagem|luggage|mala|malas|idosos?|elderly|criancas|crianças|kids|children)\b",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def _requested_route_option_modes(user_message: str) -> set[str]:
    """Return transport modes explicitly requested as route options."""
    normalized = _normalize_token(user_message)
    modes: set[str] = set()
    if re.search(r"\bmetro\b", normalized):
        modes.add("metro")
    if re.search(r"\b(?:bus|buses|autocarro|autocarros)\b", normalized):
        modes.add("bus")
    if re.search(r"\b(?:tram|trams|eletrico|eletricos|electrico|electricos)\b", normalized):
        modes.add("tram")
    if re.search(r"\b(?:cp|comboio|comboios|train|trains)\b", normalized):
        modes.add("train")
    return modes


def _query_strictly_requests_train_only(user_message: str) -> bool:
    """Return whether the user explicitly forbids non-train legs."""
    normalized = _normalize_token(user_message)
    return bool(
        re.search(
            r"\b(?:only|just|apenas|s[oó])\b(?:\s+de)?\s+(?:cp|comboio|comboios|train|trains)\b|"
            r"\b(?:cp|comboio|comboios|train|trains)\s+(?:only|apenas)\b",
            normalized,
        )
    )


def _build_mode_unavailable_response(
    *,
    mode: str,
    origin: str,
    destination: str,
    language: str,
) -> str:
    """Build an honest answer when an explicitly requested mode cannot cover a route."""
    timestamp = datetime.now().strftime("%H:%M")
    origin_display = _get_transport_display_name(origin)
    destination_display = _get_transport_display_name(destination)

    if mode == "metro":
        if language == "pt":
            return "\n".join(
                [
                    f"### 🚇 **{origin_display} → {destination_display}**",
                    "",
                    "✅ **Resposta direta:** não consegui confirmar uma rota **de Metro** para esta ligação.",
                    "",
                    "---",
                    "",
                    "- 🚇 O Metro de Lisboa só cobre estações da própria rede; pelo menos um dos pontos pedidos não ficou confirmado como estação de Metro.",
                    "- Não vou substituir por autocarro, CP ou ferry porque pediste especificamente **metro**.",
                    "- Uma alternativa multimodal suportada pode combinar Metro, CP ou autocarro quando os operadores tiverem dados para a rota.",
                    "",
                    f"📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Atualizado:** {timestamp}",
                ]
            ).strip()
        return "\n".join(
            [
                f"### 🚇 **{origin_display} → {destination_display}**",
                "",
                "✅ **Direct answer:** I could not confirm a **Metro** route for this trip.",
                "",
                "---",
                "",
                "- 🚇 Metro de Lisboa only covers its own station network; at least one requested point was not confirmed as a Metro station.",
                "- I will not replace it with bus, CP, or ferry because you specifically asked for **Metro**.",
                "- A supported multimodal alternative may combine Metro, CP, or bus when those operators have route data.",
                "",
                f"📌 **Source:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Updated:** {timestamp}",
            ]
        ).strip()

    if language == "pt":
        return "\n".join(
            [
                f"### 🚆 **{origin_display} → {destination_display}**",
                "",
                f"✅ **Resposta direta:** não consegui confirmar uma rota apenas de **{mode}** para esta ligação nos dados disponíveis.",
                "",
                "- Uma alternativa multimodal suportada pode combinar acesso por Metro ou autocarro a uma estação CP e depois CP suburbana/AML, quando os dados confirmarem a ligação.",
                "",
                f"📌 **Fonte:** [*CP*](https://www.cp.pt) | **Atualizado:** {timestamp}",
            ]
        ).strip()
    return "\n".join(
        [
            f"### 🚆 **{origin_display} → {destination_display}**",
            "",
            f"✅ **Direct answer:** I could not confirm a **{mode} only** route for this trip from the available data.",
            "",
            "- A supported multimodal alternative may combine Metro or bus access to a CP station and then CP suburban/AML rail when the data confirms the connection.",
            "",
            f"📌 **Source:** [*CP*](https://www.cp.pt) | **Updated:** {timestamp}",
        ]
    ).strip()


def _is_generic_public_transport_route_query(user_message: str) -> bool:
    """Return whether a route asks for public transport without choosing a specific mode."""
    normalized = _normalize_token(user_message)
    if _query_has_route_mode_constraints(user_message):
        return False
    generic_markers = [
        "transportes publicos",
        "transporte publico",
        "public transport",
        "public transit",
        "transit",
    ]
    if any(marker in normalized for marker in generic_markers):
        return True
    if not _extract_route_endpoints(user_message):
        return False
    return bool(
        re.search(
            r"\b(?:como\s+(?:posso\s+)?(?:ir|vou|chego)|quero\s+(?:ir|chegar)|"
            r"preciso\s+(?:de\s+)?ir|qual\s+(?:e\s+)?(?:a\s+)?(?:melhor|forma)|"
            r"how\s+(?:do\s+i\s+)?(?:get|go)|want\s+to\s+(?:go|get)|"
            r"best\s+(?:way|option))\b",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def _strip_transport_source_lines(text: str) -> str:
    """Remove per-tool source footers before composing multi-mode route sections."""
    lines = []
    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        if re.match(
            r"^\s*(?:📌|ðŸ“Œ)?\s*\*\*(?:Fontes?|Sources?):",
            stripped,
            flags=re.IGNORECASE,
        ):
            continue
        lines.append(raw_line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _strip_embedded_transport_route_block(text: str) -> str:
    """Remove wrapper title/direct-answer lines before embedding a route block."""
    kept: List[str] = []
    skipping_embedded_alternatives = False
    for raw_line in _strip_transport_source_lines(text).splitlines():
        stripped = raw_line.strip()
        if re.search(
            r"\*\*(?:Outras opções que também fazem sentido|Other sensible options):\*\*",
            stripped,
            flags=re.IGNORECASE,
        ):
            skipping_embedded_alternatives = True
            continue
        if skipping_embedded_alternatives:
            if not stripped:
                skipping_embedded_alternatives = False
            continue
        if re.match(r"^###\s+", stripped):
            continue
        if re.match(r"^✅\s+\*\*(?:Resposta direta|Direct answer):\*\*", stripped, flags=re.IGNORECASE):
            continue
        if stripped == "---" and not any(line.strip() for line in kept):
            continue
        kept.append(raw_line.rstrip())
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and kept[-1].strip() in {"", "---"}:
        kept.pop()
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _route_result_uses_metro(route_result: str) -> bool:
    """Return whether a route result contains a Lisbon Metro leg."""
    normalized = _normalize_token(route_result)
    if "metro route" in normalized:
        return True
    return bool(
        re.search(
            r"\b(?:origin is metro|origem (?:is )?metro|trajeto de metro|"
            r"nearest metro|metro mais proximo|embarque na estacao|linha (?:azul|amarela|verde|vermelha))\b",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def _route_result_is_metro_only_partial(route_result: str) -> bool:
    """Return whether a Metro route result only reaches a nearby Metro station."""
    normalized = _normalize_token(route_result)
    return bool(
        re.search(
            r"\b(?:destination|destino)\b.{0,80}\b(?:not on metro|nao fica na rede do metro|não fica na rede do metro|not on the metro)\b",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def _bare_aml_municipality_label(value: str) -> str:
    """Return the canonical AML municipality label for an exact broad endpoint."""
    normalized = _normalize_token(value)
    if not normalized:
        return ""
    if normalized == "vila franca":
        return "Vila Franca de Xira"
    if normalized == "setubal":
        return "Setúbal"
    for municipality in AML_MUNICIPALITY_NAMES:
        if normalized == _normalize_token(municipality):
            return municipality
    return ""


def _stop_is_canonical_for_bare_municipality(stop_text: str, municipality: str) -> bool:
    """Return whether a stop looks like the canonical centre/station for a municipality."""
    normalized_stop = _normalize_token(stop_text)
    normalized_municipality = _normalize_token(municipality)
    if not normalized_stop or not normalized_municipality:
        return False
    if normalized_stop == normalized_municipality:
        return True
    if normalized_municipality not in normalized_stop:
        return False
    return bool(
        re.search(
            r"\b(?:estacao|terminal|centro|camara|municipio|municipal)\b",
            normalized_stop,
            flags=re.IGNORECASE,
        )
    )


def _build_bare_municipality_clarification(
    *,
    origin: str,
    destination: str,
    municipality: str,
    rejected_stop: str = "",
    language: str = "pt",
) -> str:
    """Ask for a precise point when a broad municipality would route to a homonym stop."""
    timestamp = datetime.now().strftime("%H:%M")
    if language == "pt":
        second_option = (
            f"- B) 🚏 **{rejected_stop}** — confirma se era esta paragem/rua concreta."
            if rejected_stop
            else "- B) 🚏 **Uma paragem/rua concreta com esse nome** — indica o nome completo."
        )
        return "\n".join(
            [
                "### 🧭 **Preciso de confirmar o destino**",
                "",
                f"✅ **Resposta direta:** **{destination}** é demasiado amplo para uma rota porta-a-porta a partir de **{origin}**.",
                "",
                "⚠️ **Ambiguidade:** posso estar a interpretar uma destas opções:",
                f"- A) 🏘️ **{municipality} município/concelho** — indica uma zona, estação, paragem ou morada dentro do concelho.",
                second_option,
                "",
                f"💡 Para uma rota precisa, escreve por exemplo: **{origin} → centro de {municipality}** ou uma morada/paragem específica dentro desse concelho.",
                "",
                f"📌 **Fonte:** [*Carris Metropolitana*](https://www.carrismetropolitana.pt) | **Atualizado:** {timestamp}",
            ]
        )

    second_option = (
        f"- B) 🚏 **{rejected_stop}** — confirm if you meant this exact stop/street."
        if rejected_stop
        else "- B) 🚏 **A specific stop/street with that name** — provide the full name."
    )
    return "\n".join(
        [
            "### 🧭 **I Need To Confirm The Destination**",
            "",
            f"✅ **Direct answer:** **{destination}** is too broad for a door-to-door route from **{origin}**.",
            "",
            "⚠️ **Ambiguity:** I may be interpreting one of these options:",
            f"- A) 🏘️ **{municipality} municipality** — specify an area, station, stop, or address inside the municipality.",
            second_option,
            "",
            f"💡 For a precise route, ask for example: **{origin} → {municipality} centre** or a specific address/stop inside that municipality.",
            "",
            f"📌 **Source:** [*Carris Metropolitana*](https://www.carrismetropolitana.pt) | **Updated:** {timestamp}",
        ]
    )


def _preferred_metropolitana_option_details(
    bus_block: str,
    destination: str,
) -> tuple[str, str]:
    """Return lines/stop from the option that best matches the requested destination."""
    if not bus_block:
        return "", ""

    option_blocks = re.split(r"(?m)^\s*-\s*🚌\s+\*\*(?:Opção|Option)\s+\d+\*\*\s*$", bus_block)
    if len(option_blocks) <= 1:
        option_blocks = [bus_block]
    else:
        option_blocks = option_blocks[1:]

    destination_norm = _normalize_token(destination)
    parsed: list[tuple[str, str, str]] = []
    for block in option_blocks:
        stop_match = re.search(r"\*\*(?:Sai em|Alight at|Get off at):\*\*\s*([^|\n]+)", block, flags=re.IGNORECASE)
        line_match = re.search(r"\*\*(?:Linhas|Lines):\*\*\s*([^\n]+)", block, flags=re.IGNORECASE)
        stop_text = (stop_match.group(1).strip() if stop_match else "").strip(".")
        line_text = (line_match.group(1).strip() if line_match else "").strip(".")
        if not stop_text and not line_text:
            continue
        stop_norm = _normalize_token(stop_text)
        parsed.append((line_text, stop_text, stop_norm))

    if not parsed:
        return "", ""

    if destination_norm:
        for line_text, stop_text, stop_norm in parsed:
            if stop_norm == destination_norm:
                return line_text, stop_text
        for line_text, stop_text, stop_norm in parsed:
            if destination_norm in stop_norm or stop_norm in destination_norm:
                return line_text, stop_text

    line_text, stop_text, _stop_norm = parsed[0]
    return line_text, stop_text


def _prioritize_metropolitana_option_blocks(bus_block: str, destination: str) -> str:
    """Move the option whose alighting stop best matches the destination first."""
    if not bus_block or not destination:
        return bus_block or ""

    option_re = re.compile(
        r"(?ms)^-\s*🚌\s+\*\*(?:Opção|Option)\s+\d+\*\*.*?(?=^\s*-\s*🚌\s+\*\*(?:Opção|Option)\s+\d+\*\*|\Z)",
    )
    matches = list(option_re.finditer(bus_block))
    if len(matches) < 2:
        return bus_block

    prefix = bus_block[: matches[0].start()]
    suffix = bus_block[matches[-1].end():]
    destination_norm = _normalize_token(destination)

    def score(block: str) -> tuple[int, int]:
        stop_match = re.search(r"\*\*(?:Sai em|Alight at|Get off at):\*\*\s*([^|\n]+)", block, flags=re.IGNORECASE)
        stop_text = (stop_match.group(1).strip() if stop_match else "").strip(".")
        stop_norm = _normalize_token(stop_text)
        if destination_norm and stop_norm == destination_norm:
            return (0, 0)
        if destination_norm and (destination_norm in stop_norm or stop_norm in destination_norm):
            return (1, 0)
        return (2, 0)

    indexed_blocks = [(idx, match.group(0)) for idx, match in enumerate(matches, start=1)]
    ordered = sorted(indexed_blocks, key=lambda item: (*score(item[1]), item[0]))
    if [idx for idx, _block in ordered] == [idx for idx, _block in indexed_blocks]:
        return bus_block

    is_pt = bool(re.search(r"\*\*Opção\s+\d+\*\*", matches[0].group(0), flags=re.IGNORECASE))
    option_label = "Opção" if is_pt else "Option"
    renumbered: list[str] = []
    for new_idx, (_old_idx, block) in enumerate(ordered, start=1):
        renumbered.append(
            re.sub(
                r"^-\s*🚌\s+\*\*(?:Opção|Option)\s+\d+\*\*",
                f"- 🚌 **{option_label} {new_idx}**",
                block,
                count=1,
                flags=re.IGNORECASE,
            ).rstrip()
        )
    prefix_clean = prefix.rstrip()
    return (f"{prefix_clean}\n" if prefix_clean else "") + "\n".join(renumbered) + suffix


def _build_metropolitana_bridge_for_partial_metro_route(
    *,
    user_message: str,
    origin: str,
    destination: str,
    route_result: str,
) -> Optional[str]:
    """Try a Metro-to-AML-bus bridge when the destination is outside Metro coverage."""
    language = _infer_language(user_message, "")
    broad_destination = _bare_aml_municipality_label(destination)
    skipped_homonym_stop = ""
    transfer_hubs = (
        "Campo Grande",
        "Senhor Roubado",
        "Oriente",
        "Sete Rios",
        "Colégio Militar/Luz",
        "Odivelas",
        "Marquês de Pombal",
        "Cais do Sodré",
    )
    try:
        from tools.carrismetropolitana_api import find_bus_routes
        from tools.transport_api import get_route_between_stations
    except Exception:
        return None

    direct_metropolitana_result = ""
    try:
        direct_metropolitana_result = str(
            find_bus_routes.invoke({"origin": origin, "destination": destination})
        ).strip()
    except Exception:
        direct_metropolitana_result = ""

    for hub in transfer_hubs:
        if _normalize_token(hub) in {_normalize_token(origin), _normalize_token(destination)}:
            continue
        try:
            hub_bus_result = str(
                find_bus_routes.invoke({"origin": hub, "destination": destination})
            ).strip()
        except Exception:
            continue
        if not hub_bus_result or _tool_result_indicates_no_match(hub_bus_result):
            continue

        metro_to_hub = _build_deterministic_metro_route_response(
            user_message=(
                f"Quero ir de {origin} para {hub} de metro."
                if language == "pt"
                else f"I want to go by metro from {origin} to {hub}."
            ),
            context="",
        )
        if not metro_to_hub:
            try:
                metro_to_hub = str(
                    get_route_between_stations.invoke({"origin": origin, "destination": hub})
                ).strip()
            except Exception:
                metro_to_hub = ""

        metro_details = _parse_route_details(metro_to_hub or route_result)
        board_station = str(metro_details.get("board_station") or "").strip()
        initial_access_note = (
            f"- 🚶 **Ligação inicial:** começa em **{origin}** e segue para a estação **{board_station}**, usada como referência de Metro mais próxima; confirma o acesso pedonal/local exato antes de sair."
            if language == "pt" and board_station and _normalize_token(board_station) != _normalize_token(origin)
            else f"- 🚶 **Initial access:** start at **{origin}** and go to **{board_station}**, used as the nearest Metro reference; confirm the exact walking/local access before leaving."
            if language != "pt" and board_station and _normalize_token(board_station) != _normalize_token(origin)
            else ""
        )

        bus_block = _localize_metropolitana_direct_bus_block(
            _clean_metropolitana_direct_bus_block(hub_bus_result),
            language,
        )
        bus_block = _prioritize_metropolitana_option_blocks(bus_block, destination)
        if not bus_block:
            continue

        line_text, stop_text = _preferred_metropolitana_option_details(bus_block, destination)
        if broad_destination and stop_text and not _stop_is_canonical_for_bare_municipality(stop_text, broad_destination):
            skipped_homonym_stop = skipped_homonym_stop or stop_text
            continue
        bus_reference = (
            f"linha(s) **{line_text}** até **{stop_text}**"
            if language == "pt" and line_text and stop_text
            else f"line(s) **{line_text}** to **{stop_text}**"
            if language != "pt" and line_text and stop_text
            else "**Carris Metropolitana** até uma paragem próxima do destino"
            if language == "pt"
            else "**Carris Metropolitana** to a stop near the destination"
        )
        final_access_note = (
            f"- 🎯 **Ponto confirmado para o destino:** sai em **{stop_text}**; o percurso final até à entrada de **{destination}** não ficou confirmado pela fonte de transportes."
            if language == "pt" and stop_text
            else f"- 🎯 **Confirmed destination-side stop:** get off at **{stop_text}**; the final access to the **{destination}** entrance is not confirmed by the transport source."
            if language != "pt" and stop_text
            else ""
        )
        realtime_note = (
            "- 📡 **Tempo real Carris Metropolitana:** esta fonte confirma linha/paragens de referência, mas não próximos autocarros nem perturbações específicas da linha."
            if language == "pt"
            else "- 📡 **Carris Metropolitana real time:** this source confirms reference lines/stops, but not next buses or line-specific disruptions."
        )
        timestamp = datetime.now().strftime("%H:%M")
        if language == "pt":
            parts = [
                f"### 🚇🚌 **{origin} → {destination}**",
                "",
                f"✅ **Resposta direta:** encontrei uma combinação suportada pelos dados: Metro até **{hub}** e depois {bus_reference}; confirma a partida antes de sair.",
                "",
                "---",
                "",
                "### 🚇 **Até ao ponto de transbordo**",
                "",
                initial_access_note,
                "",
                _strip_embedded_transport_route_block(metro_to_hub or route_result),
                "",
                "---",
                "",
                "### 🚌 **Carris Metropolitana**",
                "",
                bus_block,
                "",
                final_access_note,
                realtime_note,
                "",
                "⚠️ **Nota:** a ferramenta confirma linhas/paragens possíveis; não confirma em tempo real a partida, perturbações específicas da linha nem o acesso final exato até à entrada do destino.",
                "",
                f"📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | [*Carris Metropolitana*](https://www.carrismetropolitana.pt) | **Atualizado:** {timestamp}",
            ]
        else:
            parts = [
                f"### 🚇🚌 **{origin} → {destination}**",
                "",
                f"✅ **Direct answer:** I found a data-supported combination: Metro to **{hub}** and then {bus_reference}; confirm the departure before leaving.",
                "",
                "---",
                "",
                "### 🚇 **To the transfer point**",
                "",
                initial_access_note,
                "",
                _strip_embedded_transport_route_block(metro_to_hub or route_result),
                "",
                "---",
                "",
                "### 🚌 **Carris Metropolitana**",
                "",
                bus_block,
                "",
                final_access_note,
                realtime_note,
                "",
                "⚠️ **Note:** the tool confirms possible lines/stops; it does not confirm real-time departure, line-specific disruptions, or exact final access to the destination entrance.",
                "",
                f"📌 **Source:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | [*Carris Metropolitana*](https://www.carrismetropolitana.pt) | **Updated:** {timestamp}",
            ]
        return "\n".join(part for part in parts if part is not None).strip()

    if direct_metropolitana_result and not _tool_result_indicates_no_match(direct_metropolitana_result):
        bus_block = _localize_metropolitana_direct_bus_block(
            _clean_metropolitana_direct_bus_block(direct_metropolitana_result),
            language,
        )
        bus_block = _prioritize_metropolitana_option_blocks(bus_block, destination)
        if bus_block:
            _line_text, stop_text = _preferred_metropolitana_option_details(bus_block, destination)
            if broad_destination and stop_text and not _stop_is_canonical_for_bare_municipality(stop_text, broad_destination):
                skipped_homonym_stop = skipped_homonym_stop or stop_text
                bus_block = ""
        if bus_block:
            timestamp = datetime.now().strftime("%H:%M")
            direct_answer = (
                "✅ **Resposta direta:** encontrei uma opção da Carris Metropolitana; o Metro só ajuda até à estação mais próxima da origem."
                if language == "pt"
                else "✅ **Direct answer:** I found a Carris Metropolitana option; Metro only helps up to the closest station near the origin."
            )
            source = (
                f"📌 **Fonte:** [*Carris Metropolitana*](https://www.carrismetropolitana.pt) | **Atualizado:** {timestamp}"
                if language == "pt"
                else f"📌 **Source:** [*Carris Metropolitana*](https://www.carrismetropolitana.pt) | **Updated:** {timestamp}"
            )
            return f"### 🚌 **{origin} → {destination}**\n\n{direct_answer}\n\n---\n\n{bus_block}\n\n{source}"

    if broad_destination and skipped_homonym_stop:
        try:
            from tools.cp_api import get_cp_station_info

            has_cp_station = bool(get_cp_station_info(destination))
        except Exception:
            has_cp_station = False
        if not has_cp_station:
            return _build_bare_municipality_clarification(
                origin=origin,
                destination=destination,
                municipality=broad_destination,
                rejected_stop=skipped_homonym_stop,
                language=language,
            )

    return None


def _extract_route_minutes_for_scoring(text: str) -> Optional[int]:
    """Extract a best-effort journey duration from a route or train block."""
    if not text:
        return None

    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    patterns = [
        r"(?:Tempo total estimado|Estimated total time|Duracao|Duration):\s*\**~?\s*(\d+)",
        r"\((\d+)\s*min\)",
        r"\b(\d+)\s*min(?:utos?)?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                continue
    return None


def _build_cp_bridge_for_partial_metro_route(
    *,
    user_message: str,
    origin: str,
    destination: str,
    route_result: str,
    requested_bus: bool = False,
    checked_metropolitana_direct: bool = False,
) -> Optional[str]:
    """Try a Metro-to-CP bridge when the destination is outside Metro coverage."""
    language = _infer_language(user_message, "")
    preferences = _parse_route_mode_preferences(user_message)
    if (
        preferences["metro_only"]
        or preferences["bus_only"]
        or preferences["tram_only"]
        or preferences["exclude_metro"]
    ):
        return None
    normalized_request = _normalize_token(user_message)
    if re.search(
        r"\b(?:sem|evitar|evito|n[aã]o\s+quero|without|avoid|no)\s+(?:cp|comboio|comboios|train|trains)\b",
        normalized_request,
    ):
        return None

    try:
        from tools.cp_api import get_cp_station_info, plan_train_trip
    except Exception:
        return None

    destination_cp = get_cp_station_info(destination)
    if not destination_cp:
        return None

    destination_station = str(destination_cp.get("name") or destination).strip()
    if not destination_station:
        return None

    cp_access_hubs: tuple[tuple[str, str], ...] = (
        ("Oriente", "Oriente"),
        ("Entrecampos", "Entrecampos"),
        ("Sete Rios", "Jardim Zoológico"),
        ("Rossio", "Rossio"),
        ("Cais do Sodré", "Cais do Sodré"),
        ("Santa Apolónia", "Santa Apolónia"),
    )

    candidates: list[dict[str, Any]] = []
    for cp_station, metro_station in cp_access_hubs:
        if _normalize_token(cp_station) == _normalize_token(destination_station):
            continue
        try:
            raw_train = str(
                plan_train_trip.invoke(
                    {"origin": cp_station, "destination": destination_station}
                )
            ).strip()
        except Exception:
            continue
        if not raw_train or _tool_result_indicates_no_match(raw_train):
            continue

        metro_prompt = (
            f"Quero ir de metro de {origin} para {metro_station}"
            if language == "pt"
            else f"I want to go by metro from {origin} to {metro_station}"
        )
        metro_block = _build_deterministic_metro_route_response(
            user_message=metro_prompt,
            context=f"User language: {language}",
        )
        if not metro_block or _route_result_is_metro_only_partial(metro_block):
            continue

        metro_minutes = _extract_route_minutes_for_scoring(metro_block) or 999
        train_minutes = _extract_route_minutes_for_scoring(raw_train) or 999
        candidates.append(
            {
                "score": metro_minutes + train_minutes,
                "cp_station": cp_station,
                "metro_station": metro_station,
                "metro_block": metro_block,
                "train_block": raw_train,
            }
        )

    if not candidates:
        return None

    selected = min(candidates, key=lambda item: int(item["score"]))
    cp_station = str(selected["cp_station"])
    metro_station = str(selected["metro_station"])
    metro_block = _strip_embedded_transport_route_block(str(selected["metro_block"]))
    train_block = _strip_transport_source_lines(str(selected["train_block"]))
    train_block = re.sub(
        r"(?m)^###\s+🚆\s+\*\*[^*\n]+\*\*\s*\n+",
        "",
        train_block,
        count=1,
    ).strip()
    train_block = normalize_cp_no_more_trains_message(train_block, language)
    no_more_cp_today = bool(
        re.search(
            r"\b(?:Sem mais comboios hoje|No more trains today)\b",
            train_block,
            flags=re.IGNORECASE,
        )
    )

    origin_display = _get_transport_display_name(origin)
    destination_display = _get_transport_display_name(destination)
    timestamp = datetime.now().strftime("%H:%M")
    include_bus_note = requested_bus or checked_metropolitana_direct
    broad_destination = _bare_aml_municipality_label(destination)

    if language == "pt":
        title = f"### 🚇🚆 **{origin_display} → {destination_display}**"
        direct = (
            f"✅ **Resposta direta:** o percurso suportado é ir de **Metro até {metro_station}** "
            f"e depois seguir de **CP suburbana/AML** para **{destination_display}**, "
            "mas a CP não mostra mais partidas hoje para esse troço."
            if no_more_cp_today
            else (
                f"✅ **Resposta direta:** a opção confirmada é ir de **Metro até {metro_station}** "
                f"e depois apanhar a **CP suburbana/AML** para **{destination_display}**."
            )
        )
        broad_note = (
            f"ℹ️ **Nota de destino:** assumi **Estação/Centro de {broad_destination}** como referência. "
            "Se queres outro ponto do concelho, indica a morada, zona ou paragem."
            if broad_destination
            else ""
        )
        metro_title = "**🚇 Acesso à CP**"
        train_title = "**🚆 Comboio / CP**"
        sequence = (
            f"- 🚇 **{origin_display} → {metro_station}:** Metro de Lisboa.\n"
            f"- 🚆 **{cp_station} → {destination_display}:** CP suburbana/AML."
        )
        bus_note = (
            "**🚌 Autocarro**\n\n"
            "- ⚠️ Não encontrei uma linha direta de autocarro **Carris Metropolitana** confirmada para este par nos dados consultados; "
            "uma alternativa sem Metro+CP exigiria confirmar transbordos mais longos."
        )
        source_links = [
            "[*Metro de Lisboa*](https://www.metrolisboa.pt)",
            "[*CP*](https://www.cp.pt)",
        ]
        if include_bus_note:
            source_links.append("[*Carris Metropolitana*](https://www.carrismetropolitana.pt)")
        source = (
            f"📌 **Fonte:** {' | '.join(source_links)} | **Atualizado:** {timestamp}"
        )
    else:
        title = f"### 🚇🚆 **{origin_display} → {destination_display}**"
        direct = (
            f"✅ **Direct answer:** the supported route is to take **Metro to {metro_station}** "
            f"and then **CP suburban/AML rail** to **{destination_display}**, "
            "but CP shows no more departures today for that rail leg."
            if no_more_cp_today
            else (
                f"✅ **Direct answer:** the confirmed option is to take **Metro to {metro_station}** "
                f"and then **CP suburban/AML rail** to **{destination_display}**."
            )
        )
        broad_note = (
            f"ℹ️ **Destination note:** I used **{broad_destination} station/centre** as the reference. "
            "If you mean another point in the municipality, provide the address, area, or stop."
            if broad_destination
            else ""
        )
        metro_title = "**🚇 Access to CP rail**"
        train_title = "**🚆 Train / CP**"
        sequence = (
            f"- 🚇 **{origin_display} → {metro_station}:** Lisbon Metro.\n"
            f"- 🚆 **{cp_station} → {destination_display}:** CP suburban/AML rail."
        )
        bus_note = (
            "**🚌 Bus**\n\n"
            "- ⚠️ I could not confirm a direct **Carris Metropolitana** bus line for this pair in the consulted data; "
            "an option without Metro+CP would require checking longer transfers."
        )
        source_links = [
            "[*Metro de Lisboa*](https://www.metrolisboa.pt)",
            "[*CP*](https://www.cp.pt)",
        ]
        if include_bus_note:
            source_links.append("[*Carris Metropolitana*](https://www.carrismetropolitana.pt)")
        source = f"📌 **Source:** {' | '.join(source_links)} | **Updated:** {timestamp}"

    parts = [
        title,
        "",
        direct,
        broad_note,
        "",
        "---",
        "",
        "🧭 **Sequência recomendada**" if language == "pt" else "🧭 **Recommended sequence**",
        sequence,
        "",
        "---",
        "",
        metro_title,
        "",
        metro_block,
        "",
        "---",
        "",
        train_title,
        "",
        train_block,
    ]
    if include_bus_note:
        parts.extend(["", "---", "", bus_note])
    parts.extend(["", source])
    return "\n".join(part for part in parts if part is not None).strip()


def _tool_result_indicates_no_match(result: str) -> bool:
    """Detects tool outputs that mean no valid route or line was found."""
    normalized = _normalize_token(result or "")
    negative_markers = [
        "no direct carris route found",
        "could not locate",
        "could not resolve",
        "sem linhas diretas",
        "sem linha direta",
        "no direct bus routes found",
        "no direct buses",
        "no direct train service found",
        "no supported aml suburban cp trip",
        "nao consegui",
        "não consegui",
        "not found",
        "no bus stops found",
        "no carris metropolitana stops found",
        "no nearby carris metropolitana stops were found",
        "nao foram encontradas paragens",
        "não foram encontradas paragens",
    ]
    return any(marker in normalized for marker in negative_markers)


def _extract_carris_mode_section(route_result: str, section_name: str) -> str:
    """Extracts the requested BUSES or TRAMS section from Carris Urban route output."""
    lines = (route_result or "").splitlines()
    target = section_name.upper().strip()
    collecting = False
    section_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped in {"TRAMS", "BUSES"}:
            if collecting and stripped != target:
                break
            collecting = stripped == target
            if collecting:
                section_lines.append(line)
            continue

        if collecting:
            section_lines.append(line)

    while section_lines and not section_lines[-1].strip():
        section_lines.pop()

    return "\n".join(section_lines).strip()


def _carris_section_has_routes(section: str) -> bool:
    """Returns whether a Carris section contains at least one concrete route line."""
    return bool(re.search(r"^\s*\d{1,4}[A-Z]?\s*:", section or "", flags=re.MULTILINE))


def _parse_carris_route_entries(section: str) -> List[Dict[str, Any]]:
    """Parses a Carris route section into route entries with summary and detail lines."""
    entries: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for raw_line in (section or "").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped in {"BUSES", "TRAMS"} or re.fullmatch(r"[-=]{3,}", stripped):
            continue
        if re.match(r"^\.\.\.\s+and\s+\d+\s+more\s+routes?\.?$", stripped, flags=re.IGNORECASE):
            continue

        route_match = re.match(r"^(\d{1,4}[A-Z]?)\s*:\s*(.+)$", stripped)
        if route_match:
            if current:
                entries.append(current)
            current = {
                "route": route_match.group(1),
                "summary": route_match.group(2).strip(),
                "details": [],
            }
            continue

        if current:
            current["details"].append(stripped)

    if current:
        entries.append(current)

    return entries


def _localize_period_label(label: str, language: str) -> str:
    """Localizes Carris service-period labels."""
    mapping = {
        "morning": ("manhã", "morning"),
        "midday": ("meio do dia", "midday"),
        "afternoon": ("tarde", "afternoon"),
        "evening": ("fim do dia", "evening"),
        "night": ("noite", "night"),
    }
    normalized = _normalize_token(label)
    if normalized in mapping:
        return mapping[normalized][0 if language == "pt" else 1]
    return label.strip()


def _minutes_from_clock(clock_text: str) -> Optional[int]:
    """Converts HH:MM clock text into minutes since midnight."""
    match = re.match(r"^(\d{2}):(\d{2})$", clock_text.strip())
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def _summarize_carris_frequency_result(
    frequency_result: str,
    language: str,
) -> List[str]:
    """Builds concise user-facing bullets from the Carris frequency tool output."""
    if not frequency_result:
        return []

    no_service_match = re.search(
        r"No scheduled trips found for route '?(?P<route>[^']+)'? today",
        frequency_result,
        flags=re.IGNORECASE,
    )
    if no_service_match:
        if language == "pt":
            return [
                "    - **Serviço programado hoje:** não foram encontradas partidas nos dados de horário disponíveis.",
                "    - **Como confirmar:** peça-me as próximas partidas numa paragem específica desta linha ou confirme em carris.pt.",
            ]
        return [
            "    - **Scheduled service today:** no departures were found in the available timetable data.",
            "    - **How to confirm:** ask me for departures at a specific stop on this line or check carris.pt.",
        ]

    total_match = re.search(r"Total de passagens hoje:\s*(\d+)", frequency_result, flags=re.IGNORECASE)
    lines = frequency_result.splitlines()
    now_minutes = datetime.now().hour * 60 + datetime.now().minute
    selected_summary: Optional[str] = None

    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        period_match = re.match(
            r"^[🌅☀️🌤️🌙🌃]\s*([A-Za-zÀ-ÿ ]+)\s*\((\d{2}:\d{2})-(\d{2}:\d{2})\)(?::\s*(.+))?$",
            stripped,
        )
        if not period_match:
            index += 1
            continue

        label = _localize_period_label(period_match.group(1), language)
        start_clock = period_match.group(2)
        end_clock = period_match.group(3)
        inline_summary = (period_match.group(4) or "").strip()
        start_minutes = _minutes_from_clock(start_clock)
        end_minutes = _minutes_from_clock(end_clock)
        is_current_period = (
            start_minutes is not None
            and end_minutes is not None
            and start_minutes <= now_minutes <= end_minutes
        )

        candidate_summary: Optional[str] = None
        if inline_summary and "sem serviço" not in _normalize_token(inline_summary):
            if language == "pt":
                candidate_summary = f"**Serviço programado:** {label}, {inline_summary}."
            else:
                candidate_summary = f"**Scheduled service:** {label}, {inline_summary}."
        else:
            window_lines: List[str] = []
            look_ahead = index + 1
            while look_ahead < len(lines):
                next_line = lines[look_ahead].strip()
                if re.match(r"^[🌅☀️🌤️🌙🌃]", next_line):
                    break
                if next_line:
                    window_lines.append(next_line)
                look_ahead += 1

            avg_match = next(
                (
                    re.search(r"Frequência média:\s*\*\*(.+?)\*\*", line, flags=re.IGNORECASE)
                    for line in window_lines
                    if "Frequência média" in line
                ),
                None,
            )
            first_last_match = next(
                (
                    re.search(r"Primeiro:\s*([^|]+)\|\s*Último:\s*(.+)$", line, flags=re.IGNORECASE)
                    for line in window_lines
                    if "Primeiro:" in line and "Último:" in line
                ),
                None,
            )

            if avg_match:
                avg_value = avg_match.group(1).strip()
                first_last_text = ""
                if first_last_match:
                    first_trip = first_last_match.group(1).strip()
                    last_trip = first_last_match.group(2).strip()
                    if language == "pt":
                        first_last_text = f" entre {first_trip} e {last_trip}"
                    else:
                        first_last_text = f" between {first_trip} and {last_trip}"
                if language == "pt":
                    candidate_summary = f"**Frequência programada:** {label}, média de {avg_value}{first_last_text}."
                else:
                    candidate_summary = f"**Scheduled frequency:** {label}, about {avg_value}{first_last_text}."

            index = look_ahead - 1

        if not candidate_summary:
            index += 1
            continue

        if is_current_period:
            selected_summary = candidate_summary
            break
        if selected_summary is None:
            selected_summary = candidate_summary

        index += 1

    summary_lines: List[str] = []
    if total_match:
        total = total_match.group(1)
        if language == "pt":
            summary_lines.append(f"    - **Passagens programadas hoje:** {total}")
        else:
            summary_lines.append(f"    - **Scheduled departures today:** {total}")
    if selected_summary:
        summary_lines.append(f"    - {selected_summary}")

    return summary_lines


def _format_carris_route_detail(
    detail: str,
    language: str,
    route: Optional[str] = None,
    frequency_lookup: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """Formats a single Carris route detail line into nested markdown bullets."""
    stripped = (detail or "").strip()
    if not stripped:
        return []

    if re.fullmatch(r"check schedule\.?", stripped, re.IGNORECASE):
        frequency_result = frequency_lookup(route) if frequency_lookup and route else ""
        frequency_summary = _summarize_carris_frequency_result(frequency_result, language)
        if frequency_summary:
            return frequency_summary
        if language == "pt":
            return [
                "    - **Horário detalhado:** esta rota não devolveu uma paragem específica para calcular partidas exatas.",
                "    - **Como confirmar:** peça-me horários numa paragem concreta desta linha.",
            ]
        return [
            "    - **Detailed timetable:** this route result did not include a specific stop for exact departures.",
            "    - **How to confirm:** ask me for departures at a specific stop on this line.",
        ]

    next_match = re.match(r"^(?:\*\*)?Next(?:\*\*)?:\s*(.+)$", stripped, re.IGNORECASE)
    if next_match:
        detail_text = next_match.group(1).strip()
        stop_match = re.search(r"\(stop\s+(.+)\)\s*$", detail_text, re.IGNORECASE)
        stop_name = stop_match.group(1).strip() if stop_match else ""
        times_text = re.sub(r"\s*\(stop\s+(.+)\)\s*$", "", detail_text, flags=re.IGNORECASE).strip()
        lines: List[str] = []
        if times_text:
            if language == "pt":
                lines.append(f"    - **Próximas partidas:** {times_text}")
            else:
                lines.append(f"    - **Next departures:** {times_text}")
        if stop_name:
            if language == "pt":
                lines.append(f"    - **Paragem:** {stop_name}")
            else:
                lines.append(f"    - **Stop:** {stop_name}")
        return lines

    stops_match = re.match(
        r"^Stops:\s*board at\s+(?P<board>.+?);\s*leave at(?:\s+stop)?\s+(?P<leave>.+?)\.?$",
        stripped,
        flags=re.IGNORECASE,
    )
    if stops_match:
        board = stops_match.group("board").strip()
        leave = stops_match.group("leave").strip()
        if language == "pt":
            return [f"    - 🚏 **Paragens:** apanha em **{board}**; sai em **{leave}**."]
        return [f"    - 🚏 **Stops:** board at **{board}**; alight at **{leave}**."]

    final_walk_match = re.match(
        r"^(?:Final walk|Caminhada final):\s*~\s*(?P<minutes>\d+)\s*min\s+(?:to|até(?:\s+ao)?)\s+(?P<destination>.+?)\.?$",
        stripped,
        flags=re.IGNORECASE,
    )
    if final_walk_match:
        minutes = final_walk_match.group("minutes")
        destination = final_walk_match.group("destination").strip()
        if re.fullmatch(r"(?:destination|destino)", destination, flags=re.IGNORECASE):
            destination = "o destino" if language == "pt" else "the destination"
        if language == "pt":
            return [f"    - 🚶 **Caminhada final:** ~{minutes} min até {destination}."]
        return [f"    - 🚶 **Final walk:** ~{minutes} min to {destination}."]

    if re.search(r"no upcoming departures were confirmed today at the matched origin stop", stripped, flags=re.IGNORECASE):
        if language == "pt":
            return [
                "    - ℹ️ **Próximas partidas:** não há partidas confirmadas hoje na paragem de origem encontrada.",
            ]
        return [
            "    - ℹ️ **Next departures:** no departures were confirmed today at the matched origin stop.",
        ]

    more_routes_match = re.match(r"^\.\.\.\s+and\s+(\d+)\s+more\s+routes?\.?$", stripped, flags=re.IGNORECASE)
    if more_routes_match:
        count = int(more_routes_match.group(1))
        if language == "pt":
            noun = "rota" if count == 1 else "rotas"
            return [f"    - ℹ️ E mais {count} {noun} encontrada{'s' if count != 1 else ''} nos dados da Carris."]
        noun = "route" if count == 1 else "routes"
        return [f"    - ℹ️ And {count} more {noun} found in the Carris data."]

    if re.search(r"real-time departure details are unavailable at this stop", stripped, flags=re.IGNORECASE):
        if language == "pt":
            return [
                "    - **Próximas partidas:** não há partidas em tempo real confirmadas para a paragem de origem encontrada.",
                "    - **Validação:** a ligação direta existe nos dados da Carris; confirma a partida pouco antes de sair.",
            ]
        return [
            "    - **Next departures:** no real-time departures were confirmed for the matched origin stop.",
            "    - **Validation:** the direct Carris connection exists in the data; confirm the departure shortly before leaving.",
        ]

    travel_match = re.match(r"^~\s*(\d+)\s*min(?:\s*travel)?$", stripped, re.IGNORECASE)
    if travel_match:
        travel_text = f"~{travel_match.group(1)} min"
        if language == "pt":
            return [f"    - **Tempo estimado:** {travel_text}"]
        return [f"    - **Estimated travel time:** {travel_text}"]

    return [f"    - {stripped}"]


def _format_carris_mode_section_markdown(
    section: str,
    language: str,
    frequency_lookup: Optional[Callable[[str], str]] = None,
    max_entries: int = 3,
    prefer_confirmed_departures: bool = True,
) -> str:
    """Formats a Carris BUSES/TRAMS section into clean markdown bullets."""
    entries = _parse_carris_route_entries(section)
    if not entries:
        return ""

    def has_confirmed_departures(entry: Dict[str, Any]) -> bool:
        """Return whether the route entry includes concrete departure evidence."""
        return any(
            re.match(r"^(?:\*\*)?Next(?:\*\*)?:\s*.+", str(detail).strip(), flags=re.IGNORECASE)
            for detail in entry.get("details", [])
        )

    def travel_minutes(entry: Dict[str, Any]) -> Optional[int]:
        """Extract the estimated in-vehicle travel time for route ranking."""
        for detail in entry.get("details", []):
            match = re.match(
                r"^~\s*(\d+)\s*min(?:\s*travel)?$",
                str(detail).strip(),
                flags=re.IGNORECASE,
            )
            if match:
                return int(match.group(1))
        return None

    def next_departure_delta_minutes(entry: Dict[str, Any]) -> int:
        """Return minutes until the first listed departure, or a large fallback."""
        now = datetime.now()
        current_minutes = now.hour * 60 + now.minute
        for detail in entry.get("details", []):
            next_match = re.match(
                r"^(?:\*\*)?Next(?:\*\*)?:\s*(?P<body>.+)$",
                str(detail).strip(),
                flags=re.IGNORECASE,
            )
            if not next_match:
                continue
            time_match = re.search(r"\b(?P<hour>\d{1,2}):(?P<minute>\d{2})\b", next_match.group("body"))
            if not time_match:
                continue
            departure_minutes = int(time_match.group("hour")) * 60 + int(time_match.group("minute"))
            delta = departure_minutes - current_minutes
            if delta < 0:
                delta += 24 * 60
            return delta
        return 9999

    def route_sort_key(entry: Dict[str, Any]) -> tuple[int, int, int, str]:
        """Prefer actionable near-term departures before pure in-vehicle duration."""
        return (
            0 if has_confirmed_departures(entry) else 1,
            next_departure_delta_minutes(entry),
            travel_minutes(entry) or 999,
            str(entry.get("route") or ""),
        )

    ordered_entries = sorted(entries, key=route_sort_key)
    if prefer_confirmed_departures and any(has_confirmed_departures(entry) for entry in ordered_entries):
        ordered_entries = [entry for entry in ordered_entries if has_confirmed_departures(entry)]
    if max_entries > 0:
        omitted_count = max(0, len(ordered_entries) - max_entries)
        ordered_entries = ordered_entries[:max_entries]
    else:
        omitted_count = 0

    blocks: List[str] = []
    for entry in ordered_entries:
        block_lines = [f"- **{entry['route']}**: {entry['summary']}"]
        for detail in entry.get("details", []):
            block_lines.extend(
                _format_carris_route_detail(
                    detail,
                    language,
                    route=entry.get("route"),
                    frequency_lookup=frequency_lookup,
                )
            )
        blocks.append("\n".join(block_lines))

    if omitted_count:
        if language == "pt":
            noun = "opção" if omitted_count == 1 else "opções"
            blocks.append(f"- ℹ️ Mais {omitted_count} {noun} com menor evidência imediata foram omitidas para manter a resposta acionável.")
        else:
            noun = "option" if omitted_count == 1 else "options"
            blocks.append(f"- ℹ️ {omitted_count} more lower-confidence {noun} omitted to keep the answer actionable.")

    return "\n\n".join(blocks).strip()


def _summarize_recommended_carris_option(markdown: str, language: str) -> str:
    """Build a direct recommendation from the first formatted Carris option."""
    if not markdown:
        return ""

    first_block = markdown.split("\n\n", 1)[0]
    route_match = re.search(r"^-\s+\*\*(?P<route>[^*]+)\*\*:\s*(?P<direction>[^\n]+)", first_block)
    if not route_match:
        return ""

    route = route_match.group("route").strip()
    direction = route_match.group("direction").strip()
    stops_match = re.search(
        r"\*\*(?:Paragens|Stops):\*\*\s*(?:apanha em|board at)\s+\*\*(?P<board>[^*]+)\*\*;\s*(?:sai em|alight at)\s+\*\*(?P<leave>[^*]+)\*\*",
        first_block,
        flags=re.IGNORECASE,
    )
    departures_match = re.search(
        r"\*\*(?:Próximas partidas|Next departures):\*\*\s*(?P<departures>[^\n]+)",
        first_block,
        flags=re.IGNORECASE,
    )
    travel_match = re.search(
        r"\*\*(?:Tempo estimado|Estimated travel time):\*\*\s*(?P<travel>[^\n]+)",
        first_block,
        flags=re.IGNORECASE,
    )
    final_walk_match = re.search(
        r"\*\*(?:Caminhada final|Final walk):\*\*\s*(?P<walk>[^\n]+)",
        first_block,
        flags=re.IGNORECASE,
    )

    if language == "pt":
        summary = f"- ✅ **Melhor opção confirmada:** apanha o **{route}** ({direction})"
        if stops_match:
            summary += (
                f" em **{stops_match.group('board').strip()}** "
                f"e sai em **{stops_match.group('leave').strip()}**"
            )
        if travel_match:
            summary += f" · **{travel_match.group('travel').strip()}**"
        if final_walk_match:
            walk_text = final_walk_match.group("walk").strip().rstrip(".")
            summary += f" + caminhada final **{walk_text}**"
        summary += "."
        if departures_match:
            summary += f"\n- 🕐 **Próximas partidas:** {departures_match.group('departures').strip()}"
        summary += "\n- 📡 **Tempo real:** próximas partidas confirmadas; sem alerta operacional específico."
        return summary

    summary = f"- ✅ **Best confirmed option:** take **{route}** ({direction})"
    if stops_match:
        summary += (
            f" at **{stops_match.group('board').strip()}** "
            f"and alight at **{stops_match.group('leave').strip()}**"
        )
    if travel_match:
        summary += f" · **{travel_match.group('travel').strip()}**"
    if final_walk_match:
        walk_text = final_walk_match.group("walk").strip().rstrip(".")
        summary += f" + final walk **{walk_text}**"
    summary += "."
    if departures_match:
        summary += f"\n- 🕐 **Next departures:** {departures_match.group('departures').strip()}"
    summary += "\n- 📡 **Real-time:** upcoming departures confirmed; no specific operational alert reported."
    return summary


def _count_formatted_carris_options(markdown: str) -> int:
    """Count route option cards in formatted Carris markdown."""
    if not markdown:
        return 0
    return len(re.findall(r"(?m)^\s*-\s+\*\*[^*\n]{1,16}\*\*:", markdown))


def _build_mode_filtered_carris_route_response(
    route_result: str,
    user_message: str,
    origin: str,
    destination: str,
) -> str:
    """Return a Carris route response filtered to the user's requested mode."""
    preferences = _parse_route_mode_preferences(user_message)
    language = _infer_language(user_message, "")
    if not (
        preferences["bus_only"]
        or preferences["tram_only"]
        or preferences["exclude_bus"]
        or preferences["exclude_tram"]
    ):
        return route_result

    mode_key = "TRAMS" if preferences["tram_only"] or preferences["exclude_bus"] else "BUSES"
    is_tram = mode_key == "TRAMS"
    selected_block = _extract_carris_mode_section(route_result, mode_key)
    other_block = _extract_carris_mode_section(route_result, "BUSES" if is_tram else "TRAMS")
    selected_markdown = _format_carris_mode_section_markdown(
        selected_block,
        language,
        max_entries=3,
    )
    if not selected_markdown:
        if language == "pt":
            requested = "elétrico" if is_tram else "autocarro"
            other = "autocarro" if is_tram else "elétrico"
            other_note = (
                f" Só surgiram opções de {other}, que omiti porque não correspondem ao modo pedido."
                if _carris_section_has_routes(other_block)
                else ""
            )
            return (
                f"### {'🚋' if is_tram else '🚌'} {origin} → {destination}\n\n"
                f"❌ **Resposta direta:** não consegui confirmar uma opção apenas de {requested} "
                f"para este trajeto nos dados disponíveis da Carris Urban.{other_note}\n\n"
                f"📌 **Fonte:** [*Carris*](https://www.carris.pt) | **Atualizado:** {datetime.now().strftime('%H:%M')}"
            )
        requested = "tram" if is_tram else "bus"
        other = "bus" if is_tram else "tram"
        other_note = (
            f" I omitted {other} options because they do not match the requested mode."
            if _carris_section_has_routes(other_block)
            else ""
        )
        return (
            f"### {'🚋' if is_tram else '🚌'} {origin} → {destination}\n\n"
            f"❌ **Direct answer:** I couldn't confirm a {requested}-only option "
            f"for this trip in the available Carris Urban data.{other_note}\n\n"
            f"📌 **Source:** [*Carris*](https://www.carris.pt) | **Updated:** {datetime.now().strftime('%H:%M')}"
        )

    recommended = _summarize_recommended_carris_option(selected_markdown, language)
    if language == "pt":
        mode_label = "Elétrico" if is_tram else "Autocarro"
        title = f"### {'🚋' if is_tram else '🚌'} **{origin} → {destination}**"
        option_count = _count_formatted_carris_options(selected_markdown)
        asks_other_same_mode = bool(
            re.search(
                r"\b(?:outros?|outras?|mais)\s+(?:autocarros?|el[eé]tricos?|linhas?)\b",
                _normalize_token(user_message),
            )
        )
        if asks_other_same_mode and option_count <= 1:
            direct = (
                f"✅ **Resposta direta:** só encontrei **uma opção de {mode_label.lower()}** "
                "confirmada para este trajeto nos dados disponíveis da Carris; "
                "não vou inventar outras linhas."
            )
        else:
            direct = (
                f"✅ **Resposta direta:** encontrei opções de **{mode_label.lower()}** "
                f"para este trajeto nos dados disponíveis da Carris."
            )
        section = f"**{'🚋' if is_tram else '🚌'} {mode_label}s**"
        source = f"📌 **Fonte:** [*Carris*](https://www.carris.pt) | **Atualizado:** {datetime.now().strftime('%H:%M')}"
    else:
        mode_label = "Tram" if is_tram else "Bus"
        title = f"### {'🚋' if is_tram else '🚌'} **{origin} → {destination}**"
        option_count = _count_formatted_carris_options(selected_markdown)
        asks_other_same_mode = bool(
            re.search(
                r"\b(?:another|other|more)\s+(?:bus|buses|tram|trams|lines?)\b",
                _normalize_token(user_message),
            )
        )
        if asks_other_same_mode and option_count <= 1:
            direct = (
                f"✅ **Direct answer:** I found only **one confirmed {mode_label.lower()} option** "
                "for this trip in the available Carris data; I will not invent other lines."
            )
        else:
            direct = (
                f"✅ **Direct answer:** I found **{mode_label.lower()}** options "
                f"for this trip in the available Carris data."
            )
        section = "**🚋 Trams**" if is_tram else "**🚌 Buses**"
        source = f"📌 **Source:** [*Carris*](https://www.carris.pt) | **Updated:** {datetime.now().strftime('%H:%M')}"

    parts = [title, "", direct]
    include_recommended = bool(recommended) and not (asks_other_same_mode and option_count <= 1)
    if include_recommended:
        parts.extend(["", recommended])
    parts.extend(["", "---", "", section, "", selected_markdown, "", source])
    return "\n".join(parts).strip()


def _build_carris_surface_route_response(
    route_result: str,
    user_message: str,
    origin: str,
    destination: str,
) -> str:
    """Return a clean Carris Urban surface route answer."""
    language = _infer_language(user_message, "")
    preferences = _parse_route_mode_preferences(user_message)
    if (
        preferences["bus_only"]
        or preferences["tram_only"]
        or preferences["exclude_bus"]
        or preferences["exclude_tram"]
    ):
        return _build_mode_filtered_carris_route_response(
            route_result,
            user_message,
            origin,
            destination,
        )

    tram_markdown = _format_carris_mode_section_markdown(
        _extract_carris_mode_section(route_result, "TRAMS"),
        language,
        max_entries=2,
    )
    bus_markdown = _format_carris_mode_section_markdown(
        _extract_carris_mode_section(route_result, "BUSES"),
        language,
        max_entries=2,
    )
    if not tram_markdown and not bus_markdown:
        return route_result

    timestamp = datetime.now().strftime("%H:%M")
    is_pt = language == "pt"
    title = f"### 🚌🚋 **{origin} → {destination}**"
    alternative_requested = bool(
        re.search(
            r"\b(?:alternativas?|op[cç][oõ]es|options|autocarro|autocarros|bus|buses|"
            r"el[eé]trico|eletrico|tram|trams)\b",
            user_message,
            flags=re.IGNORECASE,
        )
    )
    direct = (
        f"✅ **Resposta direta:** encontrei uma opção direta na Carris Urban entre **{origin}** e **{destination}**."
        if is_pt and not alternative_requested
        else f"✅ **Resposta direta:** encontrei opções de superfície na Carris Urban entre **{origin}** e **{destination}**."
        if is_pt
        else f"✅ **Direct answer:** I found a direct Carris Urban option between **{origin}** and **{destination}**."
        if not alternative_requested
        else f"✅ **Direct answer:** I found Carris Urban surface options between **{origin}** and **{destination}**."
    )
    source = (
        f"📌 **Fonte:** [*Carris*](https://www.carris.pt) | **Atualizado:** {timestamp}"
        if is_pt
        else f"📌 **Source:** [*Carris*](https://www.carris.pt) | **Updated:** {timestamp}"
    )
    parts = [title, "", direct, "", "---", ""]
    if tram_markdown:
        parts.extend([
            "### 🚋 **Elétrico**" if is_pt else "### 🚋 **Tram**",
            "",
            tram_markdown,
            "",
        ])
    if bus_markdown:
        parts.extend([
            "### 🚌 **Autocarros**" if is_pt else "### 🚌 **Buses**",
            "",
            bus_markdown,
            "",
        ])
    parts.append(source)
    return "\n".join(parts).strip()


def _clean_metropolitana_direct_bus_block(text: str) -> str:
    """Removes wrapper lines from Carris Metropolitana direct-bus output for embedding in summaries."""
    cleaned_lines: List[str] = []
    for raw_line in (text or "").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"[-=]{3,}", stripped):
            continue
        if (
            stripped.startswith("🚌 **Autocarros:")
            or stripped.startswith("🚌 **Buses:")
            or re.match(r"^#{1,6}\s+🚌\s+\*\*Carris Metropolitana:", stripped)
        ):
            continue
        if "linha(s) direta(s) encontrada(s):" in stripped or "direct line(s) found:" in stripped:
            continue
        if stripped.startswith("💡 **Como usar:") or stripped.startswith("💡 **How to use"):
            break
        if stripped.startswith(("⚠️ **Note:**", "⚠️ **Nota:**", "💡 **Tips:**", "💡 **Dicas:**")):
            break
        if re.search(r"\bCheck carrismetropolitana\.pt\b|\bMetro may be faster\b", stripped, flags=re.IGNORECASE):
            break
        if stripped.startswith("📌 **Fonte:") or stripped.startswith("📌 **Source:"):
            continue
        if stripped.startswith("⚠️ Scope:") or stripped.startswith("💡 For Lisbon city-only"):
            continue
        cleaned_lines.append(raw_line)

    return _linkify_metropolitana_coordinate_suffixes("\n".join(cleaned_lines).strip())


def _linkify_metropolitana_coordinate_suffixes(text: str) -> str:
    """Replace internal coordinate suffixes with compact Google Maps links."""
    if not text:
        return text

    def repl(match: re.Match[str]) -> str:
        lat = match.group("lat")
        lon = match.group("lon")
        return f" | [Paragem](https://www.google.com/maps/search/?api=1&query={lat}%2C{lon})"

    return re.sub(
        r"\s*\|\s*(?:coords|coordinates|coordenadas)\s*:\s*(?P<lat>-?\d+(?:\.\d+)?),\s*(?P<lon>-?\d+(?:\.\d+)?)",
        repl,
        text,
        flags=re.IGNORECASE,
    )


def _localize_metropolitana_direct_bus_block(text: str, language: str) -> str:
    """Localizes retained Carris Metropolitana direct-route lines."""
    if language != "pt" or not text:
        return text

    localized = text
    localized = re.sub(
        r"\b(\d+)\s+direct route option\(s\) found\b",
        lambda match: (
            f"{match.group(1)} opção direta encontrada"
            if match.group(1) == "1"
            else f"{match.group(1)} opções diretas encontradas"
        ),
        localized,
        flags=re.IGNORECASE,
    )
    localized = re.sub(
        r"\b(\d+)\s+Line match\(es\)",
        lambda match: (
            f"{match.group(1)} linha compatível"
            if match.group(1) == "1"
            else f"{match.group(1)} linhas compatíveis"
        ),
        localized,
        flags=re.IGNORECASE,
    )
    replacements = [
        (r"\bOption\s+(\d+)\b", r"Opção \1"),
        (r"\*\*Alight at:\*\*", "**Sai em:**"),
        (r"\*\*Board at:\*\*", "**Apanha em:**"),
        (r"\*\*Lines:\*\*", "**Linhas:**"),
    ]
    for pattern, replacement in replacements:
        localized = re.sub(pattern, replacement, localized, flags=re.IGNORECASE)
    return _linkify_metropolitana_coordinate_suffixes(localized)


def _extract_metropolitana_nearby_lines(text: str, location_label: str) -> Optional[str]:
    """Extracts the nearby-line inventory shown for one endpoint in a no-direct-line result."""
    if not text or not location_label:
        return None

    match = re.search(
        rf"At\s+{re.escape(location_label)}:\s*(?P<lines>.+)$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return None

    return match.group("lines").strip()


def _format_metropolitana_no_direct_summary(
    text: str,
    *,
    origin: str,
    destination: str,
    language: str,
    build_source_line: Callable[[str, List[str]], str],
) -> str:
    """Formats a concise no-direct-line summary for Carris Metropolitana results."""
    title = (
        f"### 🚌 Linhas diretas da Carris Metropolitana para {origin} → {destination}"
        if language == "pt"
        else f"### 🚌 Direct Carris Metropolitana lines for {origin} → {destination}"
    )
    summary_lines = [
        "- ❌ **Sem linha suburbana direta confirmada** para esta ligação."
        if language == "pt"
        else "- ❌ **No direct suburban bus line was confirmed** for this trip."
    ]

    origin_lines = _extract_metropolitana_nearby_lines(text, origin)
    destination_lines = _extract_metropolitana_nearby_lines(text, destination)
    if origin_lines:
        summary_lines.append(
            f"- 📍 **Linhas disponíveis perto da origem:** {origin_lines}"
            if language == "pt"
            else f"- 📍 **Lines available near the origin:** {origin_lines}"
        )
    if destination_lines:
        summary_lines.append(
            f"- 📍 **Linhas disponíveis perto do destino:** {destination_lines}"
            if language == "pt"
            else f"- 📍 **Lines available near the destination:** {destination_lines}"
        )

    scope_markers = [
        "important: carris metropolitana scope note",
        "for lisbon city-only trips",
        "lisbon city",
    ]
    if any(marker in _normalize_token(text) for marker in scope_markers):
        summary_lines.append(
            "- ⚠️ **Âmbito:** para trajetos dentro da cidade de Lisboa, confirma também a Carris ou uma combinação Metro + autocarro."
            if language == "pt"
            else "- ⚠️ **Scope:** for Lisbon city-only trips, also confirm Carris Urban or a Metro + bus combination."
        )

    return "\n".join(
        [
            title,
            "",
            *summary_lines,
            "",
            build_source_line(
                language,
                ["[*Carris Metropolitana*](https://www.carrismetropolitana.pt)"],
            ),
        ]
    ).strip()


def _parse_wait_targets_from_route(route_result: str) -> List[Tuple[str, str]]:
    """Parses the stations and directions needed for live metro wait times."""
    details = _parse_route_details(route_result)
    directions = details.get("directions", [])
    targets: List[Tuple[str, str]] = []

    if details.get("board_station") and directions:
        targets.append((details["board_station"], directions[0]))

    if details.get("transfer_station") and len(directions) >= 2:
        targets.append((details["transfer_station"], directions[1]))

    deduped: List[Tuple[str, str]] = []
    seen = set()
    for station, direction in targets:
        key = (_normalize_token(station), _normalize_token(direction))
        if key not in seen:
            seen.add(key)
            deduped.append((station, direction))

    return deduped


def _parse_wait_targets_from_response(response: str) -> List[Tuple[str, str]]:
    """Fallback parser for station and direction pairs from the model response."""
    targets: List[Tuple[str, str]] = []
    directions = re.findall(r"direção\s+\**([^*\n\(]+)", response, flags=re.IGNORECASE)

    origin_match = re.search(
        r"Embarque na estação[: ]*\**(?P<station>[^*\n\(]+)",
        response,
        flags=re.IGNORECASE,
    )
    transfer_match = re.search(
        r"Transfer[êe]ncia em[: ]*\**(?P<station>[^*\n\(]+)",
        response,
        flags=re.IGNORECASE,
    )

    if origin_match and directions:
        targets.append((origin_match.group("station").strip(), directions[0].strip()))
    if transfer_match and len(directions) >= 2:
        targets.append((transfer_match.group("station").strip(), directions[1].strip()))

    return targets


def _upsert_realtime_wait_section(response: str, lines: List[str], language: str = "pt") -> str:
    """Replaces or inserts the real-time section before the tip or source block."""
    if not lines:
        return response

    response_lines = response.splitlines()
    section_start = None
    section_end = None

    for idx, line in enumerate(response_lines):
        stripped = line.strip()
        if stripped.startswith("🗓️") and "Próximos Metros" in stripped:
            section_start = idx
            continue
        if section_start is not None and (stripped.startswith("💡") or stripped.startswith("📌")):
            section_end = idx
            break

    if section_start is None:
        section_start = len(response_lines)
        for idx, line in enumerate(response_lines):
            stripped = line.strip()
            if stripped.startswith("💡") or stripped.startswith("📌"):
                section_start = idx
                break
        section_end = section_start
    elif section_end is None:
        section_end = len(response_lines)

    section_title = "🗓️ **Próximos Metros** (tempo real):" if language == "pt" else "🗓️ **Next Metros** (real time):"
    section_lines = [section_title, *lines, ""]
    new_lines = response_lines[:section_start] + section_lines + response_lines[section_end:]
    return "\n".join(new_lines).strip()


def _infer_language(user_message: str, context: str) -> str:
    """Infers response language from context and the user message."""
    if "PT-PT" in context or "Portuguese" in context:
        return "pt"
    if "English" in context:
        return "en"

    return infer_response_language(user_query=user_message, default="en")


def _is_future_transport_planning_query(user_message: str) -> bool:
    """Returns whether the transport query is clearly about future planning.

    Args:
        user_message: Original transport query.

    Returns:
        bool: True when the query targets a future day or planning horizon and
        should therefore avoid live wait/departure times.
    """
    normalized = _normalize_token(user_message)
    if not normalized:
        return False

    for pattern, date_format in (
        (r"\b(202\d-\d{2}-\d{2})\b", "%Y-%m-%d"),
        (r"\b(\d{2}/\d{2}/202\d)\b", "%d/%m/%Y"),
    ):
        match = re.search(pattern, user_message)
        if not match:
            continue
        try:
            explicit_date = datetime.strptime(match.group(1), date_format).date()
        except ValueError:
            continue
        if explicit_date > datetime.now().date():
            return True

    future_patterns = [
        r"\btomorrow\b",
        r"\bamanha\b",
        r"\bamanh\S*\b",
        r"\bthis weekend\b",
        r"\bweekend\b",
        r"\bfim de semana\b",
        r"\bnext week\b",
        r"\bproxima semana\b",
        r"\bnext month\b",
        r"\bproximo mes\b",
        r"\b(?:2|3|4|5)\s+days?\b",
        r"\b(?:2|3|4|5)\s+dias?\b",
        r"\bday\s+[23]\b",
        r"\b[23](?:o|a)?\s+dia\b",
        r"\bplan\b",
        r"\bplanning\b",
        r"\broteiro\b",
        r"\bitinerario\b",
        r"\bitinerary\b",
    ]
    return any(re.search(pattern, normalized) for pattern in future_patterns)


def _build_future_realtime_note(language: str) -> str:
    """Builds a localized note explaining why live waits were omitted.

    Args:
        language: Preferred response language.

    Returns:
        str: Localized future-planning note.
    """
    if language == "pt":
        return (
            "- 🗓️ **Planeamento futuro:** omiti tempos em tempo real porque esta pergunta é para outro momento. "
            "Confirme esperas e partidas no próprio dia em metrolisboa.pt."
        )
    return (
        "- 🗓️ **Future planning:** I omitted real-time waits because this request is for a later trip. "
        "Check live waits and departures on the day itself at metrolisboa.pt."
    )


def _build_future_metro_wait_limit_response(
    station: str,
    direction: Optional[str],
    language: str,
) -> str:
    """Builds a localized response for future metro wait-time queries.

    Args:
        station: Requested station.
        direction: Optional direction constraint.
        language: Preferred response language.

    Returns:
        str: Localized response explaining that live wait times are not shown for
        future planning.
    """
    title = (
        f"🚇 **{station}** → **{direction}**"
        if direction
        else (
            f"🚇 **Próximo metro em {station}**"
            if language == "pt"
            else f"🚇 **Next metro at {station}**"
        )
    )
    source_label = "📌 **Fonte:**" if language == "pt" else "📌 **Source:**"
    updated_label = "**Atualizado:**" if language == "pt" else "**Updated:**"

    if language == "pt":
        note = (
            "- ⚠️ **Tempo real indisponível para planeamento futuro:** os tempos de espera do Metro só são úteis no próprio momento da viagem."
        )
    else:
        note = (
            "- ⚠️ **Real-time waits are not shown for future planning:** Metro wait times are only meaningful close to the trip itself."
        )

    return "\n".join(
        [
            title,
            "",
            note,
            _build_future_realtime_note(language),
            "",
            f"{source_label} [*Metro de Lisboa*](https://www.metrolisboa.pt) | {updated_label} {datetime.now().strftime('%H:%M')}",
        ]
    ).strip()


def _query_requests_future_cp_schedule(user_message: str) -> bool:
    """Return whether a CP query asks for future departures/schedules.

    CP tools in this runtime expose current/near-term suburban departures; they
    do not accept a target date/time. This guard prevents live departures from
    being presented as if they were tomorrow's timetable.
    """
    normalized = _normalize_token(user_message)
    if not normalized or not _is_future_transport_planning_query(user_message):
        return False
    if not re.search(r"\b(?:cp|comboio|comboios|train|trains|rail|railway)\b", normalized):
        return False
    return bool(
        re.search(
            r"\b(?:horarios?|hor.rios?|schedule|timetable|partidas?|saidas?|departures?|proxim[ao]s?|next|apanhar|catch|ir|go)\b",
            normalized,
        )
    )


def _build_future_cp_schedule_limit_response(user_message: str, language: str) -> str:
    """Build a scoped response for future CP schedule requests."""
    endpoints = _extract_route_endpoints(user_message)
    if endpoints:
        title = f"### 🚆 **{endpoints[0]} → {endpoints[1]}**"
    else:
        title = "### 🚆 **Horários CP futuros**" if language == "pt" else "### 🚆 **Future CP schedules**"

    if language == "pt":
        body = [
            title,
            "",
            "✅ **Resposta direta:** não vou mostrar partidas em tempo real como se fossem horários de amanhã ou de uma data futura.",
            "",
            "---",
            "",
            "- 🗓️ **Limite dos dados:** a ferramenta CP disponível no LISBOA confirma partidas próximas/atuais, mas não recebe uma data e hora futura específicas.",
            "- 🚆 **O que posso fazer agora:** indicar a lógica da ligação suburbana/AML e, no próprio dia, confirmar as próximas partidas.",
            "- 🔎 **Para uma resposta precisa:** pergunta no dia da viagem ou indica que queres apenas a rota, sem horários em tempo real.",
            "",
            f"📌 **Fonte:** [*CP*](https://www.cp.pt) | **Atualizado:** {datetime.now().strftime('%H:%M')}",
        ]
    else:
        body = [
            title,
            "",
            "✅ **Direct answer:** I will not show live departures as if they were tomorrow's or a future-date timetable.",
            "",
            "---",
            "",
            "- 🗓️ **Data limit:** LISBOA's confirmed CP coverage supports current/near-term departures, but it does not accept a specific future date and time.",
            "- 🚆 **What I can do now:** explain the suburban/AML route logic and confirm next departures on the travel day.",
            "- 🔎 **For an exact answer:** ask on the day of travel or ask only for the route, without real-time times.",
            "",
            f"📌 **Source:** [*CP*](https://www.cp.pt) | **Updated:** {datetime.now().strftime('%H:%M')}",
        ]
    return "\n".join(body).strip()


def _query_requests_broad_carris_catalog(user_message: str) -> bool:
    """Return whether the user asks for an impractically broad Carris live dump."""
    normalized = _normalize_token(user_message)
    if not normalized:
        return False
    has_carris_context = bool(re.search(r"\b(?:carris|autocarros?|buses|paragens?|stops|linhas?|lines)\b", normalized))
    if not has_carris_context:
        return False
    asks_all_catalog = bool(
        re.search(
            r"\b(?:todas?|todos?|all|every)\b.*\b(?:linhas?|lines|paragens?|stops)\b"
            r"|\b(?:linhas?|lines|paragens?|stops)\b.*\b(?:todas?|todos?|all|every)\b",
            normalized,
        )
    )
    asks_live = bool(re.search(r"\b(?:tempo real|real[-\s]?time|live|agora|now)\b", normalized))
    return asks_all_catalog and asks_live


def _build_broad_carris_catalog_limit_response(language: str) -> str:
    """Build a scoped response for all-lines/all-stops Carris live requests."""
    if language == "pt":
        return "\n".join(
            [
                "### 🚌 **Carris em tempo real**",
                "",
                "✅ **Resposta direta:** não é útil nem fiável despejar todas as linhas e todas as paragens da Carris numa só resposta em tempo real.",
                "",
                "---",
                "",
                "- 🧭 **Como posso responder com qualidade:** diz-me uma linha, paragem, zona ou origem → destino.",
                "- 🚌 **Exemplos válidos:** `próximo 758 em Amoreiras`, `autocarro de Avenidas Novas para Campo de Ourique`, `chegadas na paragem Rossio`.",
                "",
                f"📌 **Fonte:** [*Carris*](https://www.carris.pt) | **Atualizado:** {datetime.now().strftime('%H:%M')}",
            ]
        ).strip()
    return "\n".join(
        [
            "### 🚌 **Carris real-time data**",
            "",
            "✅ **Direct answer:** dumping every Carris line and every stop in one live answer is not useful or reliable.",
            "",
            "---",
            "",
            "- 🧭 **How I can answer well:** give me a line, stop, area, or origin → destination.",
            "- 🚌 **Good examples:** `next 758 at Amoreiras`, `bus from Avenidas Novas to Campo de Ourique`, `arrivals at Rossio stop`.",
            "",
            f"📌 **Source:** [*Carris*](https://www.carris.pt) | **Updated:** {datetime.now().strftime('%H:%M')}",
        ]
    ).strip()


def _join_transport_labels(labels: List[str], language: str) -> str:
    """Join transport-scope labels into a user-facing list."""
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    separator = " e " if language == "pt" else " and "
    return ", ".join(labels[:-1]) + separator + labels[-1]


def _detect_unsupported_transport_modes(user_message: str) -> List[str]:
    """Detect transport modes that the current runtime does not verify directly."""
    normalized = _normalize_token(user_message)
    if not normalized:
        return []

    def _mode_negated(mode_pattern: str) -> bool:
        return bool(
            re.search(
                rf"\b(?:nao\s+quero|sem|evitar|evita|avoid|no)\s+(?:ir\s+de\s+|usar\s+|apanhar\s+|use\s+)?(?:{mode_pattern})\b",
                normalized,
            )
        )

    supported_network_hint = bool(
        re.search(
            r"\b(metro|carris|autocarro|autocarros|bus|buses|comboio|comboios|train|trains|cp|tram|trams|eletrico|eletricos|electrico|electricos)\b",
            normalized,
        )
    )
    supported_network_negated = bool(
        re.search(
            r"\b(?:nao\s+quero|sem|evitar|evita|avoid|no)\s+(?:ir\s+de\s+|usar\s+|apanhar\s+|use\s+)?"
            r"(?:metro|carris|autocarro|autocarros|bus|buses|comboio|comboios|train|trains|cp|tram|trams|eletrico|eletricos|electrico|electricos)\b",
            normalized,
        )
    )
    ride_hailing_detail_hint = bool(
        re.search(
            r"\b(price|cost|fare|estimate|estimated|eta|wait time|waiting time|how much|quanto custa|preco|preço|tarifa|espera|tempo de espera)\b",
            normalized,
        )
    )

    comparison_hint = bool(
        re.search(r"\b(?:mais\s+rapido|rapido|faster|fastest|better|melhor|comparar|compare)\b", normalized)
    )

    unsupported_modes: List[str] = []

    def _append(mode_code: str) -> None:
        if mode_code not in unsupported_modes:
            unsupported_modes.append(mode_code)

    if re.search(r"\b(ferry|ferries|boat|boats|barco|barcos|transtejo|soflusa|cacilheiro|cacilheiros)\b", normalized):
        _append("ferries")

    if re.search(r"\bfertagus\b", normalized):
        _append("fertagus")

    if (
        re.search(r"\b(cp|comboio|comboios|train|trains)\b", normalized)
        and _CP_LONG_DISTANCE_DESTINATION_RE.search(normalized)
    ):
        _append("long_distance_cp")

    if (
        re.search(
            r"\b(gira|bike|bikes|bicycle|bicycles|bicicleta|bicicletas|scooter|scooters|e-scooter|e-scooters|e scooter|e scooters|trotinete|trotinetes)\b",
            normalized,
        )
        and not _mode_negated(
            r"gira|bike|bikes|bicycle|bicycles|bicicleta|bicicletas|scooter|scooters|e-scooter|e-scooters|e scooter|e scooters|trotinete|trotinetes"
        )
        and (comparison_hint or supported_network_negated or not supported_network_hint)
    ):
        _append("micromobility")

    if re.search(r"\b(uber|bolt|taxi|taxis|táxi|táxis)\b", normalized) and (
        not _mode_negated(r"uber|bolt|taxi|taxis")
        and (ride_hailing_detail_hint or comparison_hint or supported_network_negated or not supported_network_hint)
    ):
        _append("ride_hailing")

    return unsupported_modes


def _build_unsupported_transport_scope_response(
    user_message: str,
    language: str,
) -> Optional[str]:
    """Build an honest limitation note for unsupported transport networks or modes."""
    unsupported_modes = _detect_unsupported_transport_modes(user_message)
    if not unsupported_modes:
        return None

    labels_en = {
        "ferries": "Transtejo/Soflusa ferries",
        "fertagus": "Fertagus trains",
        "long_distance_cp": "long-distance CP services outside the Lisbon Metropolitan Area",
        "micromobility": "Gira bikes or shared e-scooters",
        "ride_hailing": "ride-hailing or taxi pricing details",
    }
    labels_pt = {
        "ferries": "ferries Transtejo/Soflusa",
        "fertagus": "comboios Fertagus",
        "long_distance_cp": "serviços CP de longo curso fora da Área Metropolitana de Lisboa",
        "micromobility": "bicicletas Gira ou trotinetes partilhadas",
        "ride_hailing": "preços ou tempos de espera de Uber, Bolt ou táxi",
    }
    labels = labels_pt if language == "pt" else labels_en
    unsupported_label = _join_transport_labels(
        [labels[mode] for mode in unsupported_modes if mode in labels],
        language,
    )
    supported_scope_pt = "Metro, Carris, Carris Metropolitana e CP suburbanos/AML"
    supported_scope_en = "Metro, Carris Urban, Carris Metropolitana, and CP suburban/AML services"
    is_long_distance_only = unsupported_modes == ["long_distance_cp"]
    unsupported_sources = {
        "ferries": "[*Transtejo/Soflusa*](https://ttsl.pt)",
        "fertagus": "[*Fertagus*](https://www.fertagus.pt)",
        "long_distance_cp": "[*CP*](https://www.cp.pt)",
        "micromobility": "[*Gira*](https://www.gira-bicicletasdelisboa.pt)",
    }
    source_tokens = [unsupported_sources[mode] for mode in unsupported_modes if mode in unsupported_sources]
    source_link = " | ".join(dict.fromkeys(source_tokens))
    timestamp = datetime.now().strftime("%H:%M")
    if unsupported_modes == ["ride_hailing"]:
        if language == "pt":
            return "\n".join(
                [
                    "### 🚕 Mobilidade fora do âmbito confirmado",
                    "",
                    "Não consigo dizer se Uber ou Bolt é melhor agora, porque este sistema não confirma preços, tempos de espera ou disponibilidade desses serviços em tempo real.",
                    "",
                    "**Para decidir no momento:**",
                    "- Abre as duas apps e compara o preço final, o tempo estimado de recolha e o tempo total antes de aceitar.",
                    "- Se preferires dados verificados pelo sistema, reformula a viagem com Metro, Carris, Carris Metropolitana ou CP suburbanos/AML.",
                ]
            ).strip()
        return "\n".join(
            [
                "### 🚕 Mobility outside confirmed scope",
                "",
                "I can't say whether Uber or Bolt is better right now, because this system does not verify real-time ride-hailing prices, estimated pickup times, or availability.",
                "",
                "**To decide in the moment:**",
                "- Open both apps and compare final price, estimated pickup time, and total trip time before accepting.",
                "- If you want system-verified data, rephrase the trip with Metro, Carris Urban, Carris Metropolitana, or CP suburban/AML services.",
            ]
        ).strip()

    if language == "pt":
        if not is_long_distance_only:
            return "\n".join(
                [
                    "### ⚠️ **Rede Fora do Âmbito Confirmado**",
                    "",
                    f"✅ **Resposta direta:** não consigo confirmar {unsupported_label} em tempo real neste sistema.",
                    "",
                    f"- O LISBOA valida diretamente **{supported_scope_pt}**.",
                    "- Posso ajudar se reformulares a viagem com uma dessas redes suportadas.",
                    "",
                    f"📌 **Fonte:** {source_link} | **Atualizado:** {timestamp}" if source_link else "",
                ]
            ).strip()
        return "\n".join(
            [
                "### 🚆 **Comboios CP Fora do Âmbito AML**",
                "",
                f"- Não consigo confirmar {unsupported_label} neste sistema.",
                f"- O LISBOA valida diretamente {supported_scope_pt}.",
                "- Para Alfa Pendular, Intercidades ou outros serviços de longo curso, confirma horários e bilhetes diretamente na CP.",
                "",
                f"📌 **Fonte:** {source_link} | **Atualizado:** {timestamp}" if source_link else "",
            ]
        ).strip()

    if not is_long_distance_only:
        lines: list[str] = [
            "### ⚠️ **Network Outside Confirmed Scope**",
            "",
            f"✅ **Direct answer:** I can't verify {unsupported_label} in real time in this system.",
            "",
            f"- LISBOA directly validates **{supported_scope_en}**.",
            "- I can help if you rephrase the trip using one of those supported networks.",
        ]
        if "ride_hailing" in unsupported_modes:
            lines.extend(
                [
                    "",
                    "- If you are deciding now, compare final price, estimated pickup time, and total trip time across ride-hailing apps.",
                ]
            )
        lines.append("")
        lines.append(f"📌 **Source:** {source_link} | **Updated:** {timestamp}" if source_link else "")
        return "\n".join(lines).strip()

    return "\n".join(
        [
            "### 🚆 **CP Trains Outside AML Scope**",
            "",
            f"- I can't directly verify {unsupported_label} with the transport data currently available in LISBOA.",
            f"- LISBOA directly validates {supported_scope_en}.",
            "- If you are deciding now, compare final price, estimated pickup time, and total trip time across ride-hailing apps."
            if "ride_hailing" in unsupported_modes
            else "",
            "- For Alfa Pendular, Intercidades, or other long-distance rail services, confirm schedules and tickets directly with CP.",
            "",
            f"📌 **Source:** {source_link} | **Updated:** {timestamp}" if source_link else "",
        ]
    ).strip()


def _build_lisbon_setubal_scope_response(user_message: str, language: str) -> Optional[str]:
    """Return an honest scope answer for common Lisbon to Setubal requests."""
    normalized = _normalize_token(user_message)
    if not re.search(r"\bsetubal\b", normalized):
        return None
    if not re.search(r"\b(lisbon|lisboa|from lisbon|de lisboa)\b", normalized):
        return None

    timestamp = datetime.now().strftime("%H:%M")
    if language == "pt":
        return "\n".join(
            [
                "### 🚆 Lisboa → Setúbal",
                "",
                "Não consigo confirmar uma rota completa Lisboa → Setúbal com os dados de transporte atualmente disponíveis no LISBOA.",
                "",
                "**Confirmado no sistema:**",
                "- A CP é usada para serviços suburbanos suportados na AML, incluindo a Linha do Sado entre Barreiro e Setúbal.",
                "",
                "**Não confirmado aqui:**",
                "- Fertagus e ferries Transtejo/Soflusa não têm verificação operacional direta nos dados atuais do LISBOA.",
                "- Não devo transformar estes operadores em horários, tempos de chegada ao vivo ou uma rota completa sem dados próprios.",
                "",
                "Para uma viagem real, confirma a travessia do Tejo e o operador final nos canais oficiais antes de sair.",
                "",
                f"📌 **Fonte:** [*CP*](https://www.cp.pt) | **Atualizado:** {timestamp}",
            ]
        ).strip()

    return "\n".join(
        [
            "### 🚆 Lisbon → Setúbal",
            "",
            "I can't confirm a complete Lisbon → Setúbal route with the transport data currently available in LISBOA.",
            "",
            "**Confirmed in the system:**",
            "- LISBOA uses CP for supported suburban AML rail data, including the Sado line between Barreiro and Setúbal.",
            "",
            "**Not confirmed here:**",
            "- Fertagus and Transtejo/Soflusa ferries are not directly verified by the current LISBOA transport data.",
            "- I should not turn those operators into live times, arrival estimates, or a complete route without their own data.",
            "",
            "For a real trip, verify the Tagus crossing and final operator on the official channels before leaving.",
            "",
            f"📌 **Source:** [*CP*](https://www.cp.pt) | **Updated:** {timestamp}",
        ]
    ).strip()


def _get_line_id_between(station_a: Optional[str], station_b: Optional[str]) -> Optional[str]:
    """Returns the shared metro line between two stations, if any."""
    if not station_a or not station_b:
        return None

    from tools.metrolisboa_api import get_station_lines

    a_lines = set(get_station_lines(station_a))
    b_lines = set(get_station_lines(station_b))
    shared = list(a_lines & b_lines)
    return shared[0] if shared else None


def _parse_route_details(route_result: str) -> dict:
    """Parses route details from the deterministic route tool output."""
    board_station_match = re.search(r"Board at\s+\*\*(?P<station>[^*]+)\*\*", route_result)
    exits = re.findall(r"Exit at\s+\*\*([^*]+)\*\*", route_result)
    transfer_station_match = re.search(r"Transfer at\*\*:\s*(?P<station>[^\n\(]+)", route_result)
    directions = re.findall(r"direção\s+\*\*([^*]+)\*\*", route_result, flags=re.IGNORECASE)
    estimated_time_match = re.search(r"Estimated travel time:\s+\*\*([^*]+)\*\*", route_result)
    walk_match = re.search(r"Walk to\s+([^\n]+)", route_result)

    return {
        "board_station": board_station_match.group("station").strip() if board_station_match else None,
        "final_station": exits[-1].strip() if exits else None,
        "transfer_station": transfer_station_match.group("station").strip() if transfer_station_match else None,
        "directions": [direction.strip() for direction in directions],
        "estimated_time": estimated_time_match.group(1).strip() if estimated_time_match else None,
        "walk_target": walk_match.group(1).strip() if walk_match else None,
    }


def _line_display_name(line_id: str, language: str) -> str:
    """Returns localized metro line labels."""
    names_pt = {
        "amarela": "Linha Amarela",
        "azul": "Linha Azul",
        "verde": "Linha Verde",
        "vermelha": "Linha Vermelha",
    }
    names_en = {
        "amarela": "Yellow Line",
        "azul": "Blue Line",
        "verde": "Green Line",
        "vermelha": "Red Line",
    }
    return (names_pt if language == "pt" else names_en).get(line_id, line_id.title())


def _looks_like_acronym_label(text: str) -> bool:
    """Returns whether the label looks like an acronym that should keep its casing."""
    stripped = (text or "").strip()
    return bool(re.fullmatch(r"(?:[A-Z0-9]{2,}(?:[\s/-][A-Z0-9]{2,})*)", stripped))


def _get_transport_display_name(location: Optional[str], detailed: bool = False) -> str:
    """Returns a user-facing location label while preserving landmark branding like NOVA IMS."""
    from tools.location_resolver import get_location_display_name
    from tools.metrolisboa_api import get_landmark_info

    raw = str(location or "").strip()
    if not raw:
        return raw

    landmark = get_landmark_info(raw)
    if landmark:
        if detailed:
            return str(
                landmark.get("display_name")
                or landmark.get("name")
                or landmark.get("short_name")
                or raw
            ).strip()
        return str(
            landmark.get("short_name")
            or landmark.get("name")
            or landmark.get("display_name")
            or raw
        ).strip()

    if _looks_like_acronym_label(raw):
        return raw

    try:
        resolved_label = get_location_display_name(raw, detailed=detailed)
        if resolved_label:
            return resolved_label
    except Exception:
        pass

    return raw.title()


def _localize_wait_times(wait_times: str, language: str) -> str:
    """Localizes small wait-time labels for display."""
    return wait_times.replace("arriving", "a chegar") if language == "pt" else wait_times


def _localize_metro_status_text(
    status: str,
    language: str,
    *,
    regular_service_open: bool = True,
) -> str:
    """Localizes common Metro status messages emitted by the official feed."""
    raw_status = str(status or "").strip()
    normalized = _normalize_token(raw_status)
    if not raw_status:
        return "estado em tempo real indisponível" if language == "pt" else "real-time status unavailable"

    if normalized == "ok":
        if not regular_service_open:
            if language == "pt":
                return "sem perturbações reportadas; fora do horário regular"
            return "no reported disruption; outside regular hours"
        return "circulação normal" if language == "pt" else "normal service"
    if normalized == "unknown":
        return "estado em tempo real indisponível" if language == "pt" else "real-time status unavailable"

    raw_interrupted_match = re.search(
        r"(?:interrupted|interrompida).*?(?:between|entre(?:\s+as\s+esta[cç][oõ]es)?)\s+(?P<start>.+?)\s+(?:and|e)\s+(?P<end>.+?)\.?$",
        raw_status,
        flags=re.IGNORECASE,
    )
    if raw_interrupted_match:
        start = raw_interrupted_match.group("start").strip(" .")
        end = raw_interrupted_match.group("end").strip(" .")
        if language == "pt":
            return f"circulação interrompida entre **{start}** e **{end}**"
        return f"service is interrupted between **{start}** and **{end}**"

    interrupted_match = re.search(
        r"circulacao esta interrompida entre as estacoes (?P<start>.+?) e (?P<end>.+?)\.?$",
        normalized,
    )
    if interrupted_match and language == "en":
        start = interrupted_match.group("start").strip().title()
        end = interrupted_match.group("end").strip().title()
        end = end.replace("Cais Do Sodre", "Cais do Sodré")
        return f"service is interrupted between **{start}** and **{end}**"

    return raw_status


def _build_route_state_lines(line_ids: List[str], language: str) -> List[str]:
    """Builds route-specific real-time line status bullets."""
    from tools.metrolisboa_api import METRO_LINES
    from tools.metrolisboa_api import get_metro_regular_service_context
    from tools.transport_api import _get_line_status

    service_context = get_metro_regular_service_context(language=language)
    regular_service_open = bool(service_context["is_regular_service_open"])
    status_lines: List[str] = []
    if not regular_service_open:
        if language == "pt":
            status_lines.append(
                "- 🌙 **Serviço ao passageiro:** fora do horário regular "
                f"({service_context['regular_window']}); confirma operação "
                "especial se vais sair agora"
            )
        else:
            status_lines.append(
                "- 🌙 **Passenger service:** outside regular operating hours "
                f"({service_context['regular_window']}); confirm special "
                "service if travelling now"
            )

    seen = set()
    for line_id in line_ids:
        if not line_id or line_id in seen:
            continue
        seen.add(line_id)
        line_info = METRO_LINES.get(line_id, {})
        emoji = line_info.get("emoji", "🚇")
        line_name = _line_display_name(line_id, language)
        status = _get_line_status(line_id)

        status_text = _localize_metro_status_text(
            status,
            language,
            regular_service_open=regular_service_open,
        )

        status_lines.append(f"- {emoji} **{line_name}**: {status_text}")

    return status_lines


def _metro_state_lines_have_disruption(state_lines: List[str]) -> bool:
    """Return whether Metro status bullets contain an actual service concern."""
    disruption_terms = (
        "interromp",
        "interrupted",
        "suspend",
        "perturb",
        "disrupt",
        "atras",
        "delay",
        "sem serviço",
        "no service",
    )
    for line in state_lines:
        normalized_line = _normalize_token(line)
        if "sem perturbacoes" in normalized_line or "no disruption" in normalized_line:
            continue
        if any(term in normalized_line for term in disruption_terms):
            return True
    return False


def _metro_state_title(state_lines: List[str], language: str, *, singular: bool = False) -> str:
    """Build a neutral or warning Metro status heading from actual line states."""
    normalized_lines = "\n".join(_normalize_token(line) for line in state_lines)
    outside_regular = bool(
        re.search(r"\b(?:fora do horario regular|outside regular)\b", normalized_lines)
    )
    icon = "⚠️" if _metro_state_lines_have_disruption(state_lines) else "🌙" if outside_regular else "🚦"
    if language == "pt":
        if outside_regular:
            label = "Horário e Estado da Linha" if singular else "Horário e Estado das Linhas"
        else:
            label = "Estado da Linha" if singular else "Estado das Linhas"
    else:
        label = "Operating Hours and Line Status" if outside_regular else "Line Status"
    return f"{icon} **{label}:**"


_GREEN_LINE_ORDER = [
    "Cais do Sodré",
    "Baixa-Chiado",
    "Rossio",
    "Martim Moniz",
    "Intendente",
    "Anjos",
    "Arroios",
    "Alameda",
    "Areeiro",
    "Roma",
    "Alvalade",
    "Campo Grande",
    "Telheiras",
]


def _green_segment_crosses_interruption(start_station: Optional[str], end_station: Optional[str]) -> bool:
    """Return whether a Green-line segment crosses the current interrupted core section."""
    if not start_station or not end_station:
        return False
    normalized_order = {_normalize_token(station): index for index, station in enumerate(_GREEN_LINE_ORDER)}
    start_index = normalized_order.get(_normalize_token(start_station))
    end_index = normalized_order.get(_normalize_token(end_station))
    if start_index is None or end_index is None:
        return False
    lower_index, upper_index = sorted([start_index, end_index])
    interruption_start = normalized_order[_normalize_token("Cais do Sodré")]
    interruption_end = normalized_order[_normalize_token("Martim Moniz")]
    return lower_index < interruption_end and upper_index > interruption_start


def _route_has_interrupted_green_segment(
    first_line_id: Optional[str],
    second_line_id: Optional[str],
    board_station: Optional[str],
    transfer_station: Optional[str],
    final_station: Optional[str],
) -> bool:
    """Detect whether a planned metro route uses the interrupted Green-line segment."""
    from tools.transport_api import _get_line_status

    if "interrompida" not in _normalize_token(_get_line_status("verde")):
        return False
    if first_line_id == "verde" and _green_segment_crosses_interruption(board_station, transfer_station or final_station):
        return True
    if second_line_id == "verde" and _green_segment_crosses_interruption(transfer_station, final_station):
        return True
    return False


def _resolve_metro_line_station_name(station_name: Optional[str]) -> Optional[str]:
    """Return the canonical Metro station name for accent-insensitive input."""
    if not station_name:
        return None
    from tools.metrolisboa_api import METRO_LINES

    station_key = _normalize_token(station_name)
    for line_info in METRO_LINES.values():
        for station in line_info.get("stations", []):
            if _normalize_token(station) == station_key:
                return station
    return None


def _format_metro_station_casing(station_name: str) -> str:
    """Returns a clean display casing for Metro station names."""
    display_name = str(station_name or "").title()
    for connector in ("De", "Do", "Da", "Dos", "Das"):
        display_name = re.sub(rf"\b{connector}\b", connector.lower(), display_name)
    return display_name


def _display_metro_line_station_name(station_name: str) -> str:
    """Return a user-facing Metro station name with proper casing and accents."""
    explicit_display_names = {
        "baixa chiado": "Baixa-Chiado",
        "baixa-chiado": "Baixa-Chiado",
        "cais do sodre": "Cais do Sodré",
        "sao sebastiao": "São Sebastião",
        "s sebastiao": "São Sebastião",
        "marques de pombal": "Marquês de Pombal",
        "praca de espanha": "Praça de Espanha",
        "santa apolonia": "Santa Apolónia",
        "terreiro do paco": "Terreiro do Paço",
    }
    normalized_station = _normalize_token(station_name)
    if normalized_station in explicit_display_names:
        return explicit_display_names[normalized_station]
    resolved_station = _resolve_metro_line_station_name(station_name)
    if resolved_station:
        return explicit_display_names.get(
            _normalize_token(resolved_station),
            _format_metro_station_casing(resolved_station),
        )
    try:
        from tools.metrolisboa_api import METRO_STATION_IDS, METRO_STATION_NAMES
    except Exception:
        return station_name.title()
    station_id = METRO_STATION_IDS.get(normalized_station) or METRO_STATION_IDS.get(str(station_name or "").lower())
    display_name = METRO_STATION_NAMES.get(station_id, str(station_name or "").title())
    return explicit_display_names.get(_normalize_token(display_name), _format_metro_station_casing(display_name))


def _display_metro_station_with_line_badges(station_name: str, language: str = "pt") -> str:
    """Return a Metro station label with all colour badges for interchange stations."""
    display_name = _display_metro_line_station_name(station_name)
    try:
        from tools.metrolisboa_api import get_station_lines

        line_ids = get_station_lines(display_name) or get_station_lines(str(station_name or ""))
    except Exception:
        line_ids = []

    line_ids = [line_id for line_id in ("amarela", "azul", "verde", "vermelha") if line_id in line_ids]
    if len(line_ids) <= 1:
        return display_name

    names_by_language = {
        "pt": {"amarela": "Amarela", "azul": "Azul", "verde": "Verde", "vermelha": "Vermelha"},
        "en": {"amarela": "Yellow", "azul": "Blue", "verde": "Green", "vermelha": "Red"},
    }
    names = names_by_language["pt"] if language == "pt" else names_by_language["en"]
    emoji_prefix = "".join(_metro_line_emoji(line_id) for line_id in line_ids)
    return f"{display_name} ({emoji_prefix} {'/'.join(names[line_id] for line_id in line_ids)})"


def _direction_for_segment(line_id: str, start_station: str, end_station: str) -> str:
    """Return the terminal direction for a segment on a Metro line."""
    from tools.metrolisboa_api import METRO_LINES

    stations = METRO_LINES.get(line_id, {}).get("stations", [])
    normalized = {_normalize_token(station): index for index, station in enumerate(stations)}
    start_index = normalized.get(_normalize_token(start_station), 0)
    end_index = normalized.get(_normalize_token(end_station), start_index)
    return stations[-1] if end_index > start_index else stations[0]


def _find_metro_path_avoiding_current_disruptions(origin: str, destination: str) -> Optional[List[Tuple[str, str, str]]]:
    """Find a Metro path while removing edges affected by known live disruptions."""
    import heapq

    from tools.metrolisboa_api import METRO_LINES

    start = _resolve_metro_line_station_name(origin)
    end = _resolve_metro_line_station_name(destination)
    if not start or not end:
        return None

    graph: Dict[str, List[Tuple[str, str]]] = {}
    for line_id, line_info in METRO_LINES.items():
        stations = line_info.get("stations", [])
        for station_a, station_b in zip(stations, stations[1:], strict=False):
            if line_id == "verde" and _green_segment_crosses_interruption(station_a, station_b):
                continue
            graph.setdefault(station_a, []).append((station_b, line_id))
            graph.setdefault(station_b, []).append((station_a, line_id))

    queue: List[Tuple[int, int, str, Optional[str], List[Tuple[str, str, str]]]] = [
        (0, 0, start, None, [])
    ]
    best_cost: Dict[Tuple[str, Optional[str]], Tuple[int, int]] = {(start, None): (0, 0)}
    while queue:
        transfers, edge_count, station, current_line, path = heapq.heappop(queue)
        if station == end:
            return path
        for next_station, line_id in graph.get(station, []):
            next_transfers = transfers + (1 if current_line and current_line != line_id else 0)
            next_edges = edge_count + 1
            state_key = (next_station, line_id)
            if best_cost.get(state_key, (10_000, 10_000)) <= (next_transfers, next_edges):
                continue
            best_cost[state_key] = (next_transfers, next_edges)
            heapq.heappush(
                queue,
                (
                    next_transfers,
                    next_edges,
                    next_station,
                    line_id,
                    [*path, (station, next_station, line_id)],
                ),
            )
    return None


def _coalesce_metro_path_segments(path: List[Tuple[str, str, str]]) -> List[Dict[str, str]]:
    """Coalesce adjacent Metro graph edges into user-facing line segments."""
    if not path:
        return []
    segments: List[Dict[str, str]] = []
    current = {"line_id": path[0][2], "start": path[0][0], "end": path[0][1]}
    for start, end, line_id in path[1:]:
        if line_id == current["line_id"] and start == current["end"]:
            current["end"] = end
            continue
        segments.append(current)
        current = {"line_id": line_id, "start": start, "end": end}
    segments.append(current)
    for segment in segments:
        segment["direction"] = _direction_for_segment(segment["line_id"], segment["start"], segment["end"])
    return segments


def _build_disruption_safe_metro_route(
    origin: str,
    destination: str,
    language: str,
    board_station: Optional[str] = None,
    final_station: Optional[str] = None,
) -> Optional[str]:
    """Build a disruption-safe alternative for any Metro route with blocked edges."""
    from tools.metrolisboa_api import METRO_LINES
    from tools.transport_api import _estimate_metro_time

    start = board_station or origin
    end = final_station or destination
    path = _find_metro_path_avoiding_current_disruptions(start, end)
    if not path:
        return None
    segments = _coalesce_metro_path_segments(path)
    if not segments:
        return None

    used_lines = [segment["line_id"] for segment in segments]
    state_lines = _build_route_state_lines([*used_lines, "verde"], language)
    wait_lines = _build_metro_wait_lines(
        [(segment["start"], segment["direction"]) for segment in segments],
        language,
    )
    estimated_time = _estimate_metro_time(len(path), transfers=max(len(segments) - 1, 0))
    timestamp = datetime.now().strftime("%H:%M")

    if language == "pt":
        lines = [
            f"### 🚇 **{_get_transport_display_name(origin)} → {_get_transport_display_name(destination)}**",
            "",
            "⚠️ **Nota de viabilidade:** a rota habitual atravessa um troço da Linha Verde interrompido. Usa a alternativa abaixo.",
            "",
            _metro_state_title(state_lines, language),
            *state_lines,
            "",
            f"⏳ **Tempo total estimado:** {estimated_time}",
            "",
            "🗺️ **Trajeto recomendado:**",
            f"- 📍 **Embarque na estação {_display_metro_line_station_name(segments[0]['start'])}**",
        ]
        for index, segment in enumerate(segments):
            if index > 0:
                lines.append(f"- 🔄 **Transferência em {_display_metro_line_station_name(segment['start'])}**")
            line_info = METRO_LINES[segment["line_id"]]
            direction = _display_metro_line_station_name(segment["direction"])
            lines.append(f"- {line_info['emoji']} **{_line_display_name(segment['line_id'], language)}** - direção **{direction}**")
        lines.extend([
            f"- 🎯 **Saia na estação {_display_metro_line_station_name(segments[-1]['end'])}**",
            "",
            "🗓️ **Próximos Metros** (tempo real):",
            *wait_lines,
            "",
            f"📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Atualizado:** {timestamp}",
        ])
        return "\n".join(lines).strip()

    lines = [
        f"### 🚇 **{_get_transport_display_name(origin)} → {_get_transport_display_name(destination)}**",
        "",
        "⚠️ **Feasibility note:** the usual route crosses an interrupted Green Line section. Use the alternative below.",
        "",
        _metro_state_title(state_lines, language),
        *state_lines,
        "",
        f"⏳ **Estimated total time:** {estimated_time}",
        "",
        "🗺️ **Recommended route:**",
        f"- 📍 **Board at {_display_metro_line_station_name(segments[0]['start'])}**",
    ]
    for index, segment in enumerate(segments):
        if index > 0:
            lines.append(f"- 🔄 **Transfer at {_display_metro_line_station_name(segment['start'])}**")
        line_info = METRO_LINES[segment["line_id"]]
        direction = _display_metro_line_station_name(segment["direction"])
        lines.append(f"- {line_info['emoji']} **{_line_display_name(segment['line_id'], language)}** - direction **{direction}**")
    lines.extend([
        f"- 🎯 **Exit at {_display_metro_line_station_name(segments[-1]['end'])}**",
        "",
        "🗓️ **Next Metros** (real time):",
        *wait_lines,
        "",
        f"📌 **Source:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Updated:** {timestamp}",
    ])
    return "\n".join(lines).strip()


def _build_practical_tip(
    language: str,
    first_direction: Optional[str],
    transfer_station: Optional[str],
    second_line_id: Optional[str],
    final_station: Optional[str],
    walk_target: Optional[str],
    walk_target_display: Optional[str] = None,
    destination_display: Optional[str] = None,
    destination_landmark: Optional[Dict[str, Any]] = None,
) -> str:
    """Builds a short, practical, non-generic travel tip."""
    if walk_target and final_station and _normalize_token(walk_target) != _normalize_token(final_station):
        walking_hint_pt = (destination_landmark or {}).get("walking_hint_pt")
        walking_hint_en = (destination_landmark or {}).get("walking_hint_en")
        metro_walk_minutes = (destination_landmark or {}).get("metro_walk_minutes")
        destination_label = destination_display or walk_target_display or walk_target

        if language == "pt":
            if walking_hint_pt:
                if metro_walk_minutes:
                    return (
                        f"Se o destino é {destination_label}, saia em {final_station} "
                        f"e conte com cerca de {metro_walk_minutes} min a pé até {walking_hint_pt}."
                    )
                return (
                    f"Se o destino é {destination_label}, saia em {final_station} "
                    f"e conte com uma caminhada final curta até {walking_hint_pt}."
                )
            return f"Da estação {final_station} até {destination_label} a caminhada final é curta."
        if walking_hint_en:
            if metro_walk_minutes:
                return (
                    f"If you're heading to {destination_label}, exit at {final_station} "
                    f"and expect about {metro_walk_minutes} minutes on foot {walking_hint_en}."
                )
            return (
                f"If you're heading to {destination_label}, exit at {final_station} "
                f"and expect a short final walk {walking_hint_en}."
            )
        return f"From {final_station} to {destination_label}, the final walk is short."

    if transfer_station and second_line_id:
        second_line_name = _line_display_name(second_line_id, language)
        if language == "pt":
            return f"Em {transfer_station}, siga a sinalização para a {second_line_name}."
        return f"At {transfer_station}, follow the signs to the {second_line_name}."

    if first_direction:
        if language == "pt":
            return f"Confirme na plataforma a direção {first_direction} antes de embarcar."
        return f"Confirm the {first_direction} direction on the platform before boarding."

    return ""


def _query_mentions_specific_route_mode(user_message: str) -> bool:
    """Returns whether the user explicitly constrained the route to a transport mode."""
    normalized = _normalize_token(user_message)
    return bool(
        re.search(
            r"\b(metro|subway|underground|comboio|comboios|train|trains|cp|autocarro|autocarros|bus|buses|tram|trams|eletrico|eletricos|electrico|electricos)\b",
            normalized,
        )
    )


def _minutes_until_clock_time(clock_text: Optional[str]) -> Optional[int]:
    """Returns minutes from now until the next HH:MM clock occurrence."""
    if not clock_text:
        return None

    target_minutes = _minutes_from_clock(clock_text)
    if target_minutes is None:
        return None

    now = datetime.now()
    current_minutes = now.hour * 60 + now.minute
    diff = target_minutes - current_minutes
    if diff < 0:
        diff += 24 * 60
    return diff


def _build_train_landmark_option(
    origin: str,
    destination: str,
    language: str,
) -> Optional[Tuple[int, str]]:
    """Builds a concise CP alternative for generic route questions when a landmark is near a rail station."""
    from tools.cp_api import get_cp_station_info, plan_train_trip
    from tools.metrolisboa_api import get_landmark_info

    origin_landmark = get_landmark_info(origin)
    destination_landmark = get_landmark_info(destination)
    if not destination_landmark:
        return None

    origin_cp = get_cp_station_info(origin)
    origin_train_station = None
    if origin_landmark and origin_landmark.get("train_station"):
        origin_train_station = str(origin_landmark["train_station"]).strip()
    elif origin_cp:
        origin_train_station = str(origin_cp.get("name") or origin).strip()

    destination_train_station = str(destination_landmark.get("train_station") or "").strip()
    if not origin_train_station or not destination_train_station:
        return None
    if _normalize_token(origin_train_station) == _normalize_token(destination_train_station):
        return None

    try:
        train_result = str(
            plan_train_trip.invoke(
                {"origin": origin_train_station, "destination": destination_train_station}
            )
        )
    except Exception:
        return None

    normalized_result = _normalize_token(train_result)
    if train_result.strip().startswith("❌") or any(
        marker in normalized_result
        for marker in [
            "no direct train service",
            "no more trains today",
            "station not found",
            "error planning trip",
        ]
    ):
        return None

    departure_match = re.search(
        r"🕐\s*\*\*(?P<departure>\d{2}:\d{2})\*\*\s*→\s*(?P<arrival>\d{2}:\d{2})\s*\((?P<travel>\d+)min\)",
        train_result,
    )
    duration_match = re.search(
        r"Dura[cç][aã]o:\s*\*\*(?P<duration>\d+(?:-\d+)?)\s*minutos\*\*",
        train_result,
        flags=re.IGNORECASE,
    )

    departure_clock = departure_match.group("departure") if departure_match else None
    travel_minutes = None
    if departure_match:
        travel_minutes = int(departure_match.group("travel"))
    elif duration_match:
        travel_minutes = int(duration_match.group("duration").split("-", 1)[0])

    walk_minutes = int(destination_landmark.get("train_walk_minutes") or 0)
    wait_minutes = _minutes_until_clock_time(departure_clock)
    score = (wait_minutes or 0) + (travel_minutes or 999) + walk_minutes

    destination_label = _get_transport_display_name(destination)
    if language == "pt":
        summary = f"- 🚆 **Comboio via {destination_train_station}**: "
        if departure_clock:
            summary += f"próxima saída às {departure_clock} desde {origin_train_station}, "
        elif origin_train_station:
            summary += f"saída desde {origin_train_station}, "
        if travel_minutes:
            summary += f"~{travel_minutes} min de viagem"
        else:
            summary += "viagem curta"
        if walk_minutes:
            summary += f" + ~{walk_minutes} min a pé até {destination_label}"
        summary += "."
    else:
        summary = f"- 🚆 **Train via {destination_train_station}**: "
        if departure_clock:
            summary += f"next departure at {departure_clock} from {origin_train_station}, "
        elif origin_train_station:
            summary += f"departing from {origin_train_station}, "
        if travel_minutes:
            summary += f"about {travel_minutes} minutes on board"
        else:
            summary += "a short rail ride"
        if walk_minutes:
            summary += f" + about {walk_minutes} minutes on foot to {destination_label}"
        summary += "."

    return score, summary


def _build_carris_direct_option(
    origin: str,
    destination: str,
    language: str,
) -> Optional[Tuple[int, str]]:
    """Builds a concise direct Carris alternative when one exists."""
    from tools.carris_api import carris_find_routes_between

    try:
        carris_result = str(
            carris_find_routes_between.invoke(
                {"origin": origin, "destination": destination}
            )
        )
    except Exception:
        return None

    normalized_result = _normalize_token(carris_result)
    if any(
        marker in normalized_result
        for marker in [
            "no direct carris route found",
            "could not locate",
            "carris database unavailable",
            "erro ao encontrar rotas",
        ]
    ):
        return None

    buses_match = re.search(
        r"BUSES\n-+\n(?P<section>.+?)(?:\n\n[A-Z]+\n-+|\Z)",
        carris_result,
        flags=re.DOTALL,
    )
    if not buses_match:
        return None

    bus_section = buses_match.group("section")
    route_match = re.search(r"^\s*(?P<route>\d{1,4}[A-Z]?): para (?P<headsign>[^\n]+)", bus_section, flags=re.MULTILINE)
    next_match = re.search(r"^\s*Next: (?P<times>[^\n]+?) \(stop (?P<stop>[^\)]+)\)", bus_section, flags=re.MULTILINE)
    travel_match = re.search(r"^\s*~(?P<travel>\d+)min travel", bus_section, flags=re.MULTILINE)
    if not route_match:
        return None

    route_code = route_match.group("route").strip()
    headsign = route_match.group("headsign").strip()
    stop_name = next_match.group("stop").strip() if next_match else None
    departure_clock = None
    if next_match:
        clock_match = re.search(r"(\d{2}:\d{2})", next_match.group("times"))
        if clock_match:
            departure_clock = clock_match.group(1)
    travel_minutes = int(travel_match.group("travel")) if travel_match else None
    wait_minutes = _minutes_until_clock_time(departure_clock)
    score = (wait_minutes or 0) + (travel_minutes or 999)

    if language == "pt":
        summary = f"- 🚌 **Autocarro {route_code}**: "
        if departure_clock:
            summary += f"próxima saída às {departure_clock}"
        else:
            summary += "rota direta disponível"
        if stop_name:
            summary += f" na paragem {stop_name}"
        if travel_minutes:
            summary += f", ~{travel_minutes} min de viagem"
        summary += f" em direção a {headsign}."
    else:
        summary = f"- 🚌 **Bus {route_code}**: "
        if departure_clock:
            summary += f"next departure at {departure_clock}"
        else:
            summary += "direct route available"
        if stop_name:
            summary += f" from stop {stop_name}"
        if travel_minutes:
            summary += f", about {travel_minutes} minutes travel"
        summary += f" toward {headsign}."

    return score, summary


def _build_additional_route_options(
    user_message: str,
    origin: str,
    destination: str,
    language: str,
) -> List[str]:
    """Builds concise alternative-mode bullets for open-ended route questions."""
    if _query_mentions_specific_route_mode(user_message):
        return []

    options: List[Tuple[int, str]] = []
    train_option = _build_train_landmark_option(origin, destination, language)
    if train_option:
        options.append(train_option)

    bus_option = _build_carris_direct_option(origin, destination, language)
    if bus_option:
        options.append(bus_option)

    if not options:
        return []

    title = (
        "🔁 **Outras opções que também fazem sentido:**"
        if language == "pt"
        else "🔁 **Other sensible options:**"
    )
    sorted_options = [summary for _, summary in sorted(options, key=lambda item: item[0])]
    return ["", title, *sorted_options, ""]


def _parse_metro_wait_request(user_message: str) -> Optional[Dict[str, Any]]:
    """Parses single-station metro wait queries with an optional direction."""
    query = user_message.strip()
    if "metro" not in query.lower():
        return None

    direction_patterns = [
        r"next metro at\s+(?P<station>.+?)\s+(?:towards|toward|to|direction)\s+(?P<direction>.+?)(?:[\?\!\.,;]|$)",
        r"when is the next metro at\s+(?P<station>.+?)\s+(?:towards|toward|to|direction)\s+(?P<direction>.+?)(?:[\?\!\.,;]|$)",
        r"pr[oó]ximo metro\s+(?:na|em)\s+(?P<station>.+?)\s+(?:sentido|dire[cç][aã]o)\s+(?P<direction>.+?)(?:[\?\!\.,;]|$)",
        r"quando passa o pr[oó]ximo metro\s+(?:na|em)\s+(?P<station>.+?)\s+(?:sentido|dire[cç][aã]o)\s+(?P<direction>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in direction_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {
                "station": _resolve_metro_station_name(match.group("station")),
                "direction": _resolve_metro_station_name(match.group("direction")),
                "status_requested": _query_has_status_intent(query),
            }

    station_only_patterns = [
        r"next metro at\s+(?P<station>.+?)(?:[\?\!\.,;]|$)",
        r"when is the next metro at\s+(?P<station>.+?)(?:[\?\!\.,;]|$)",
        r"metro wait time at\s+(?P<station>.+?)(?:[\?\!\.,;]|$)",
        r"pr[oó]ximo metro\s+(?:na|em)\s+(?P<station>.+?)(?:[\?\!\.,;]|$)",
        r"pr[oó]ximo metro\s+(?P<station>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in station_only_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {
                "station": _resolve_metro_station_name(match.group("station")),
                "direction": None,
                "status_requested": _query_has_status_intent(query),
            }

    compact_station_match = re.fullmatch(r"metro\s+(?P<station>.+)", query, flags=re.IGNORECASE)
    if compact_station_match and len(query.split()) <= 2:
        return {
            "station": _resolve_metro_station_name(compact_station_match.group("station")),
            "direction": None,
            "status_requested": _query_has_status_intent(query),
        }

    return None


def _resolve_carris_stop(stop_reference: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolves a Carris stop name into a stop ID and canonical stop label."""
    from tools.carris_api import _get_db_connection, _search_stop_rows

    conn = _get_db_connection()
    if not conn:
        return None, None

    try:
        rows = _search_stop_rows(conn, stop_reference, limit=5)
        if not rows:
            return None, None
        return rows[0]["stop_id"], rows[0]["stop_name"]
    finally:
        conn.close()


def _parse_carris_line_stop_query(user_message: str) -> Optional[Dict[str, Optional[str]]]:
    """Parses Carris wait/ETA questions for a specific line and stop."""
    query = user_message.strip()
    if not _query_has_wait_departure_intent(query):
        return None

    stop_id_match = re.search(r"\b(?:stop|paragem)\s+(?P<stop_id>\d{2,})\b", query, flags=re.IGNORECASE)

    def _line_token_match() -> Optional[re.Match[str]]:
        """Return a route-like numeric token, excluding durations or counts."""
        for match in re.finditer(r"\b(?P<line>\d{1,4}[A-Za-z]?)\b", query):
            before = query[max(0, match.start() - 35):match.start()]
            after = query[match.end():match.end() + 25]
            if re.match(r"\s*(?:min(?:utos?)?|minutes?|mins?)\b", after, flags=re.IGNORECASE):
                continue
            has_line_context = bool(
                re.search(
                    r"(?:linha|line|route|autocarro|autocarros|bus|buses|tram|"
                    r"el[eé]trico|el[eé]tricos)\s*$",
                    before,
                    flags=re.IGNORECASE,
                )
                or re.search(
                    r"^\s*(?:em|na|no|at|from|de|do|da|para|to|passa|sai|parte|"
                    r"leaves|departs)\b",
                    after,
                    flags=re.IGNORECASE,
                )
            )
            if has_line_context:
                return match
        return None

    line_match = _line_token_match()

    eta_patterns = [
        r"\beta\s+(?:for|of)\s+(?:the\s+)?(?P<line>\d{1,4}[A-Za-z]?)(?:\s+(?:bus|tram|el[eé]trico|autocarro))?\s+(?:at|em|na|no)\s+(?P<stop>.+?)(?:[\?\!\.,;]|$)",
        r"\beta\s+(?:of\s+(?:route\s+)?)?(?P<line>\d{1,4}[A-Za-z]?)(?:\s+(?:bus|tram|el[eé]trico|autocarro))?\s+(?:at|em|na|no)\s+(?P<stop>.+?)(?:[\?\!\.,;]|$)",
        r"when\s+is\s+the\s+next\s+(?:bus|tram|el[eé]trico|autocarro)?\s*(?P<line>\d{1,4}[A-Za-z]?)(?:\s+(?:bus|tram|el[eé]trico|autocarro))?\s+(?:at|em|na|no)\s+(?P<stop>.+?)(?:[\?\!\.,;]|$)",
        r"quando chega\s+(?:o\s+)?(?P<line>\d{1,4}[A-Za-z]?)\s+(?:a|ao|à|na|no|em)\s+(?P<stop>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in eta_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {
                "kind": "eta",
                "line": match.group("line").upper(),
                "stop_name": _clean_query_fragment(match.group("stop")),
                "stop_id": stop_id_match.group("stop_id") if stop_id_match else None,
            }

    if re.search(r"(?:next\s+arrivals|arrivals|pr[oó]ximas\s+chegadas|chegadas)", query, flags=re.IGNORECASE) and stop_id_match:
        return {
            "kind": "arrivals",
            "line": None,
            "stop_name": None,
            "stop_id": stop_id_match.group("stop_id"),
        }

    stop_name_arrival_patterns = [
        r"(?:quais\s+)?(?:os\s+)?pr[oó]xim(?:os|as)\s+(?:autocarros?|el[eé]tricos?|autocarros?\s+da\s+carris|ve[ií]culos?\s+da\s+carris|partidas|chegadas)\s+(?:da\s+carris\s+)?(?:at|em|na|no)\s+(?P<stop>.+?)(?:[\?\!\.,;]|$)",
        r"(?:next\s+)?(?:buses?|trams?|buses?\s+or\s+trams?|trams?\s+or\s+buses?|arrivals|departures)\s+(?:for\s+carris\s+)?(?:at|in)\s+(?P<stop>.+?)(?:[\?\!\.,;]|$)",
        r"(?:next\s+)?(?:buses?|trams?|arrivals|departures)\s+(?:for\s+carris\s+)?(?:at|in)\s+(?P<stop>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in stop_name_arrival_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {
                "kind": "arrivals",
                "line": None,
                "stop_name": _clean_query_fragment(match.group("stop")),
                "stop_id": stop_id_match.group("stop_id") if stop_id_match else None,
            }

    line_from_stop_patterns = [
        r"\b(?:pr[oó]ximas?\s+)?partidas?\s+(?:do\s+)?(?:el[eé]trico|tram|autocarro|bus)?\s*(?P<line>\d{1,4}[A-Za-z]?)\s+(?:em|no|na|at)\s+(?P<stop>.+?)(?:\s+(?:para|to)\b|[\?\!\.,;]|$)",
        r"\b(?:o|a|the)?\s*(?P<line>\d{1,4}[A-Za-z]?)\s+(?:sai|parte|passa|leaves|departs)\s+d(?:e|o|a)?\s+(?P<stop>.+?)(?:\s+(?:nos?\s+pr[oó]ximos?|next|within|agora|now)\b|[\?\!\.,;]|$)",
        r"\b(?:next|pr[oó]xim[oa])\s+(?:tram|el[eé]trico|bus|autocarro)?\s*(?P<line>\d{1,4}[A-Za-z]?)\s+(?:from|de|do|da)\s+(?P<stop>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in line_from_stop_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {
                "kind": "departures",
                "line": match.group("line").upper(),
                "stop_name": _clean_query_fragment(match.group("stop")),
                "stop_id": stop_id_match.group("stop_id") if stop_id_match else None,
            }

    stop_match = re.search(r"\b(?:at|em|na|no)\s+(?P<stop>[^\?\!\.,;]+)", query, flags=re.IGNORECASE)
    if line_match and stop_match:
        return {
            "kind": "departures",
            "line": line_match.group("line").upper(),
            "stop_name": _clean_query_fragment(stop_match.group("stop")),
            "stop_id": stop_id_match.group("stop_id") if stop_id_match else None,
        }

    return None


def _query_requests_all_metro_lines_wait_times(query: str) -> bool:
    """Detects whole-network Metro wait-time questions (no specific line mentioned).

    Examples:
        - "Qual é o tempo de espera de todas as estações do metro?"
        - "Tempos de espera em todas as linhas do metro"
        - "Wait times across all metro lines"
        - "Next metros at every station"
    """
    if _extract_metro_line_id(query):
        return False
    normalized = _normalize_token(query)
    if "metro" not in normalized and not re.search(r"\bmetro\b", query, flags=re.IGNORECASE):
        return False
    if not _query_has_wait_departure_intent(query):
        return False
    scope_patterns = [
        r"\btodas?\s+as\s+(?:\w+\s+)?estacoes\b",
        r"\btodas?\s+as\s+(?:\w+\s+)?linhas\b",
        r"\btoda\s+a\s+rede\b",
        r"\ball\s+(?:\w+\s+)?stations\b",
        r"\ball\s+(?:\w+\s+)?lines\b",
        r"\bevery\s+(?:\w+\s+)?(?:station|line)\b",
        r"\bwhole\s+network\b",
        r"\bentire\s+network\b",
    ]
    return any(re.search(pattern, normalized) for pattern in scope_patterns)


def _query_requests_top_metro_wait_times(query: str) -> bool:
    """Detects ranked Metro wait-time questions such as highest/longest waits."""
    normalized = _normalize_token(query)
    if "metro" not in normalized or not _query_has_wait_departure_intent(query):
        return False
    return bool(
        re.search(
            r"\b(?:maior(?:es)?|mais\s+(?:alt[oa]s?|long[oa]s?|demorad[oa]s?)|highest|longest|top)\b",
            normalized,
        )
    )


def _extract_top_wait_count(query: str, default: int = 3) -> int:
    """Extract a small requested result count for ranked wait-time answers."""
    normalized = _normalize_token(query)
    match = re.search(r"\b(?:top\s*)?(?P<count>\d{1,2})\s+(?:estacoes|stations|paragens|stops)\b", normalized)
    if not match:
        match = re.search(r"\b(?:top|primeir[oa]s?|first)\s+(?P<count>\d{1,2})\b", normalized)
    if not match:
        return default
    try:
        return max(1, min(int(match.group("count")), 10))
    except ValueError:
        return default


def _metro_wait_to_seconds(wait_text: str) -> Optional[int]:
    """Parse Metro wait labels into seconds for ranking."""
    normalized = _normalize_token(wait_text)
    if not normalized:
        return None
    if "arriving" in normalized or "a chegar" in normalized:
        return 0
    total = 0
    minute_match = re.search(r"(?P<minutes>\d+)\s*(?:min|m)\b", normalized)
    second_match = re.search(r"(?P<seconds>\d+)\s*s\b", normalized)
    if minute_match:
        total += int(minute_match.group("minutes")) * 60
    if second_match:
        total += int(second_match.group("seconds"))
    if total:
        return total
    plain_seconds = re.search(r"\b(?P<seconds>\d{1,4})\s*(?:segundos?|seconds?)\b", normalized)
    if plain_seconds:
        return int(plain_seconds.group("seconds"))
    return None


def _build_top_metro_wait_snapshot(user_message: str, language: str) -> str:
    """Build a ranked network-wide Metro wait-time answer."""
    from tools.metrolisboa_api import get_metro_line_wait_times

    count = _extract_top_wait_count(user_message, default=3)
    ranked: list[dict[str, Any]] = []
    for line_id in ["amarela", "azul", "verde", "vermelha"]:
        try:
            raw = str(get_metro_line_wait_times.invoke({"line": line_id}))
        except Exception:
            raw = ""
        entries, _updated_at = _parse_metro_line_wait_entries(raw)
        for entry in entries:
            station = str(entry.get("station") or "").strip()
            for direction in entry.get("directions") or []:
                wait_label = str(direction.get("wait") or "").strip()
                seconds = _metro_wait_to_seconds(wait_label)
                if seconds is None:
                    continue
                ranked.append(
                    {
                        "line_id": line_id,
                        "station": station,
                        "direction": str(direction.get("direction") or "").strip(),
                        "wait": wait_label,
                        "seconds": seconds,
                    }
                )

    title = "### 🚇 **Maiores tempos de espera no Metro**" if language == "pt" else "### 🚇 **Highest Lisbon Metro waits**"
    if not ranked:
        direct = (
            "não consegui confirmar agora um ranking de tempos de espera em tempo real no Metro."
            if language == "pt"
            else "I could not confirm a real-time Metro wait ranking right now."
        )
        return f"{title}\n\n✅ **{'Resposta direta' if language == 'pt' else 'Direct answer'}:** {direct}"

    ranked.sort(key=lambda item: item["seconds"], reverse=True)
    selected = ranked[:count]
    direct = (
        f"estas são as **{len(selected)}** maiores esperas observadas no snapshot em tempo real do Metro."
        if language == "pt"
        else f"these are the **{len(selected)}** highest waits observed in the Metro real-time snapshot."
    )
    direction_label = "sentido" if language == "pt" else "towards"
    lines = [title, "", f"✅ **{'Resposta direta' if language == 'pt' else 'Direct answer'}:** {direct}", "", "---", ""]
    for index, item in enumerate(selected, 1):
        line_display = _line_display_name(item["line_id"], language)
        line_emoji = _metro_line_emoji(item["line_id"])
        wait_label = _localize_wait_times(str(item["wait"]), language)
        lines.append(
            f"{index}. {line_emoji} **{item['station']}** — {line_display}, "
            f"{direction_label} **{item['direction']}**: **{wait_label}**"
        )
    note = (
        "💡 **Nota:** este ranking usa a maior próxima espera por plataforma no snapshot disponível e muda rapidamente."
        if language == "pt"
        else "💡 **Note:** this ranking uses the largest next-platform wait in the available snapshot and changes quickly."
    )
    footer = (
        f"📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Atualizado:** {datetime.now().strftime('%H:%M')}"
        if language == "pt"
        else f"📌 **Source:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Updated:** {datetime.now().strftime('%H:%M')}"
    )
    lines.extend(["", note, "", footer])
    return "\n".join(lines).strip()


def _query_requests_full_metro_wait_inventory(query: str) -> bool:
    """Return whether the user explicitly wants a full station-by-station dump."""
    normalized = _normalize_token(query)
    return bool(
        re.search(
            r"\b(?:lista\s+completa|listagem\s+completa|estacao\s+a\s+estacao|"
            r"todas\s+detalhadas|sem\s+resumir|full\s+list|complete\s+list|"
            r"station\s+by\s+station|every\s+station\s+in\s+detail)\b",
            normalized,
        )
    )


def _build_all_metro_lines_wait_snapshot(language: str, user_message: str = "") -> str:
    """Aggregates per-line wait snapshots into a network-wide response."""
    from tools.metrolisboa_api import get_metro_line_wait_times

    all_line_ids = ["amarela", "azul", "verde", "vermelha"]
    title = (
        "### 🚇 **Tempos de espera do Metro de Lisboa**"
        if language == "pt"
        else "### 🚇 **Lisbon Metro wait times**"
    )
    direct_answer = (
        "✅ **Resposta direta:** estes são os tempos de espera em tempo real disponíveis "
        "para todas as linhas do **Metro de Lisboa**, organizados por estação."
        if language == "pt"
        else "✅ **Direct answer:** these are the real-time wait times available across the "
        "**Lisbon Metro**, organized by station."
    )
    sections: list[str] = [title, "", direct_answer, "", "---", ""]
    try:
        from tools.metrolisboa_api import METRO_LINES
    except Exception:
        METRO_LINES = {}

    route_label = "Percurso" if language == "pt" else "Route"
    direction_prefix = "sentido" if language == "pt" else "towards"
    full_inventory = _query_requests_full_metro_wait_inventory(user_message)
    all_ranked: list[dict[str, Any]] = []
    line_summaries: list[dict[str, Any]] = []

    for line_id in all_line_ids:
        try:
            raw = str(get_metro_line_wait_times.invoke({"line": line_id}))
        except Exception:
            raw = ""

        line_name = _line_display_name(line_id, language)
        line_emoji = _metro_line_emoji(line_id)
        entries, _updated_at = _parse_metro_line_wait_entries(raw)
        line_info = METRO_LINES.get(line_id, {}) if isinstance(METRO_LINES, dict) else {}
        station_ids = list(line_info.get("stations", []))
        terminal_text = ""
        if len(station_ids) >= 2:
            terminal_text = (
                f"{_display_metro_line_station_name(station_ids[0])} ↔ "
                f"{_display_metro_line_station_name(station_ids[-1])}"
            )

        line_ranked: list[dict[str, Any]] = []
        for entry in entries:
            station = str(entry.get("station") or "").strip()
            for direction in entry.get("directions") or []:
                wait_time = _localize_wait_times(str(direction.get("wait") or ""), language)
                seconds = _metro_wait_to_seconds(wait_time)
                if seconds is None:
                    continue
                item = {
                    "line_id": line_id,
                    "line_name": line_name,
                    "station": station,
                    "direction": str(direction.get("direction") or "").strip(),
                    "wait": wait_time,
                    "seconds": seconds,
                }
                line_ranked.append(item)
                all_ranked.append(item)

        if not full_inventory:
            line_ranked.sort(key=lambda item: item["seconds"], reverse=True)
            line_summaries.append(
                {
                    "line_id": line_id,
                    "line_name": line_name,
                    "terminal_text": terminal_text,
                    "station_count": len(entries),
                    "max_item": line_ranked[0] if line_ranked else None,
                }
            )
            continue

        sections.append(f"### {line_emoji} **{line_name}**")
        if terminal_text:
            sections.append(f"- **{route_label}:** {terminal_text}")

        if not entries:
            unavailable = (
                "Tempos em tempo real indisponíveis nesta linha neste momento."
                if language == "pt"
                else "Real-time waits are unavailable on this line right now."
            )
            sections.append(f"- {unavailable}")
            sections.append("")
            continue

        for entry in entries:
            directions = entry.get("directions") or []
            if not directions:
                continue
            direction_parts = []
            for direction in directions[:2]:
                destination = direction.get("direction")
                wait_time = _localize_wait_times(str(direction.get("wait") or ""), language)
                if destination and wait_time:
                    direction_parts.append(f"{direction_prefix} {destination}: **{wait_time}**")
            if direction_parts:
                station = entry.get("station")
                sections.append(f"- **{station}:** {' | '.join(direction_parts)}")
        sections.append("")

    if not full_inventory:
        sections = [
            title,
            "",
            (
                "✅ **Resposta direta:** este é um resumo operacional dos tempos de espera em tempo real; "
                "para não tornar a resposta enorme, mostro o pior caso por linha e as maiores esperas da rede."
                if language == "pt"
                else "✅ **Direct answer:** this is an operational summary of real-time waits; "
                "to avoid a huge answer, I show the worst case by line and the highest waits network-wide."
            ),
            "",
            "---",
            "",
            "### 📊 **Resumo por linha**" if language == "pt" else "### 📊 **Line summary**",
            "",
        ]
        direction_label = "sentido" if language == "pt" else "towards"
        for item in line_summaries:
            line_emoji = _metro_line_emoji(str(item["line_id"]))
            max_item = item.get("max_item")
            if not max_item:
                unavailable = (
                    "sem tempos em tempo real confirmados agora"
                    if language == "pt"
                    else "no real-time waits confirmed right now"
                )
                sections.append(f"- {line_emoji} **{item['line_name']}**: {unavailable}.")
                continue
            route_suffix = f" · {item['terminal_text']}" if item.get("terminal_text") else ""
            if language == "pt":
                sections.append(
                    f"- {line_emoji} **{item['line_name']}**{route_suffix}: "
                    f"{item['station_count']} estações reportadas; maior espera em "
                    f"**{max_item['station']}** ({direction_label} **{max_item['direction']}**): "
                    f"**{max_item['wait']}**."
                )
            else:
                sections.append(
                    f"- {line_emoji} **{item['line_name']}**{route_suffix}: "
                    f"{item['station_count']} stations reported; highest wait at "
                    f"**{max_item['station']}** ({direction_label} **{max_item['direction']}**): "
                    f"**{max_item['wait']}**."
                )

        all_ranked.sort(key=lambda item: item["seconds"], reverse=True)
        top_items = all_ranked[:5]
        if top_items:
            sections.extend(
                [
                    "",
                    "### ⏱️ **Maiores esperas agora**" if language == "pt" else "### ⏱️ **Highest waits now**",
                    "",
                ]
            )
            for index, item in enumerate(top_items, 1):
                line_emoji = _metro_line_emoji(str(item["line_id"]))
                sections.append(
                    f"{index}. {line_emoji} **{item['station']}** — {item['line_name']}, "
                    f"{direction_label} **{item['direction']}**: **{item['wait']}**"
                )
        sections.append("")

    sections.extend(
        [
            (
                "💡 **Nota:** pede “lista completa estação a estação” se precisares do detalhe integral; estes valores mudam rapidamente."
                if language == "pt"
                else "💡 **Note:** ask for the “complete station-by-station list” if you need the full detail; these values change quickly."
            ),
            "",
        ]
    )
    return "\n".join(sections).strip()


def _build_deterministic_metro_wait_response(
    user_message: str,
    context: str,
) -> Optional[str]:
    """Builds a deterministic single-station metro wait response without losing tool detail."""
    language = _infer_language(user_message, context)
    requested_line_id = _extract_metro_line_id(user_message)

    if not requested_line_id and _query_requests_top_metro_wait_times(user_message):
        if _is_future_transport_planning_query(user_message):
            label = "rede do Metro de Lisboa" if language == "pt" else "Lisbon Metro network"
            return _build_future_metro_wait_limit_response(label, None, language)
        return _build_top_metro_wait_snapshot(user_message, language=language)

    if not requested_line_id and _query_requests_all_metro_lines_wait_times(user_message):
        if _is_future_transport_planning_query(user_message):
            label = "rede do Metro de Lisboa" if language == "pt" else "Lisbon Metro network"
            return _build_future_metro_wait_limit_response(label, None, language)
        return _build_all_metro_lines_wait_snapshot(language=language, user_message=user_message)

    if requested_line_id and _query_requests_metro_line_wait_times(user_message):
        if _is_future_transport_planning_query(user_message):
            station_label = _line_display_name(requested_line_id, language)
            return _build_future_metro_wait_limit_response(station_label, None, language)

        from tools.metrolisboa_api import get_metro_line_wait_times

        try:
            line_wait_result = str(get_metro_line_wait_times.invoke({"line": requested_line_id}))
        except Exception:
            line_wait_result = ""
        return _format_metro_line_wait_snapshot(
            line_id=requested_line_id,
            wait_result=line_wait_result,
            language=language,
            user_message=user_message,
        )

    request = _parse_metro_wait_request(user_message)
    if not request:
        return None

    future_planning = _is_future_transport_planning_query(user_message)
    station = request.get("station") or ""
    direction = request.get("direction")
    status_requested = bool(request.get("status_requested"))

    if future_planning:
        return _build_future_metro_wait_limit_response(station, direction, language)

    from tools.metrolisboa_api import get_metro_line_wait_times, get_metro_wait_time, get_station_lines

    if requested_line_id and not direction:
        try:
            line_wait_result = str(get_metro_line_wait_times.invoke({"line": requested_line_id}))
        except Exception:
            line_wait_result = ""
        state_lines = _build_route_state_lines([requested_line_id], language)
        line_name = _line_display_name(requested_line_id, language)
        line_emoji = _metro_line_emoji(requested_line_id)
        title = (
            f"### {line_emoji} **{line_name} em {station}**"
            if language == "pt"
            else f"### {line_emoji} **{line_name} at {station}**"
        )
        response_lines = [title, "", _metro_state_title(state_lines, language, singular=True), *state_lines, ""]
        if any("interrupted" in line.lower() or "interrompida" in line.lower() for line in state_lines):
            response_lines.append(
                f"- ⚠️ **Wait time:** unavailable at {station} because this section is currently interrupted."
                if language != "pt"
                else f"- ⚠️ **Tempo de espera:** indisponível em {station} porque este troço está interrompido."
            )
        else:
            station_pattern = re.compile(rf"{re.escape(station)}.*?(?P<time>\d+\s*min[^\n]*)", re.IGNORECASE)
            station_match = station_pattern.search(line_wait_result)
            if station_match:
                response_lines.append(f"- ⏱️ **Wait time:** {station_match.group('time').strip()}")
            else:
                response_lines.append(
                    "- ℹ️ **Wait time:** no station-specific live wait was returned in the line snapshot."
                    if language != "pt"
                    else "- ℹ️ **Tempo de espera:** o snapshot da linha não devolveu tempo específico para a estação."
                )
        return "\n".join(response_lines).strip()

    wait_args = {"station": station}
    if direction:
        wait_args["direction"] = direction

    try:
        wait_result = str(get_metro_wait_time.invoke(wait_args))
    except Exception:
        return None

    if wait_result.strip().startswith("❌"):
        return wait_result

    line_ids: List[str] = []
    if direction:
        shared_line = _get_line_id_between(station, direction)
        if shared_line:
            line_ids = [shared_line]
    if not line_ids:
        line_ids = list(dict.fromkeys(get_station_lines(station)))

    waits_section = "🗓️ **Próximos Metros** (tempo real):" if language == "pt" else "🗓️ **Next Metros** (real time):"
    title = (
        f"🚇 **{station}** → **{direction}**"
        if direction
        else (
            f"🚇 **Próximo metro em {station}**"
            if language == "pt"
            else f"🚇 **Next metro at {station}**"
        )
    )

    response_lines = [title]
    if status_requested or line_ids:
        state_lines = _build_route_state_lines(line_ids, language)
        response_lines.append(_metro_state_title(state_lines, language))
        response_lines.extend(state_lines)
        response_lines.append("")

    response_lines.append(waits_section)
    if direction:
        response_lines.extend(_build_metro_wait_lines([(station, direction)], language))
    else:
        blocks = _parse_metro_wait_blocks(wait_result)
        station_label = "Estação" if language == "pt" else "Station"
        direction_label = "Direção" if language == "pt" else "Direction"
        next_label = "⏱️ Próximo Metro em:" if language == "pt" else "⏱️ Next Metro in:"
        if not blocks:
            response_lines.append("- Sem dados em tempo real" if language == "pt" else "- No real-time data available")
        for block in blocks:
            localized_times = [
                _localize_wait_times(time_text, language) for time_text in block.get("times", [])
            ]
            response_lines.append(
                f"- **{station_label} {station}:** {direction_label} {block['direction']} — **{next_label}** {' | '.join(localized_times)}"
            )
            note = _localize_platform_note(block.get("note"), language)
            if note:
                response_lines.append(f"    - ℹ️ {note}")
    return "\n".join(response_lines).strip()


def _format_carris_next_departures_output(
    raw_output: str,
    *,
    line: str,
    stop_name: str,
    language: str,
) -> str:
    """Format Carris next-departure tool output with localized labels."""
    raw = str(raw_output or "").strip()
    no_more_match = re.search(r"No more departures found after\s+(?P<time>\d{1,2}:\d{2})", raw, flags=re.IGNORECASE)
    if no_more_match:
        time_text = no_more_match.group("time")
        line_label = str(line or "").upper().strip()
        stop_label = str(stop_name or "").strip() or ("a paragem indicada" if language == "pt" else "the selected stop")
        icon = "🚋" if line_label.endswith("E") else "🚌"
        if language == "pt":
            return (
                f"### {icon} **Próximas partidas do {line_label} em {stop_label}**\n\n"
                f"✅ **Resposta direta:** não encontrei mais partidas confirmadas da linha **{line_label}** "
                f"em **{stop_label}** depois das **{time_text}** nos dados disponíveis.\n\n"
                "---\n\n"
                "- ℹ️ **O que isto significa:** pode já não haver serviço listado hoje nessa paragem, ou a fonte não ter partidas futuras para esta combinação linha/paragem.\n"
                "- 💡 **Dica rápida:** confirma a paragem exata e a linha na Carris antes de sair, sobretudo em horários de menor frequência."
            )
        return (
            f"### {icon} **Next {line_label} departures at {stop_label}**\n\n"
            f"✅ **Direct answer:** I did not find more confirmed departures for line **{line_label}** "
            f"at **{stop_label}** after **{time_text}** in the available data.\n\n"
            "---\n\n"
            "- ℹ️ **What this means:** there may be no listed service left today at that stop, or the source has no future departures for this line/stop pair.\n"
            "- 💡 **Quick tip:** confirm the exact stop and line with Carris before leaving, especially during lower-frequency hours."
        )
    if not raw or "Next Departures" not in raw:
        return raw

    icon = "🚋" if str(line or "").upper().endswith("E") else "🚌"
    line_label = str(line or "").upper().strip()
    stop_label = str(stop_name or "").strip() or ("a paragem indicada" if language == "pt" else "the selected stop")
    groups: list[tuple[str, str, str]] = []
    current_route = ""
    current_destination = ""
    freshness_note = ""

    for raw_line in raw.splitlines():
        stripped = raw_line.strip().rstrip()
        if not stripped or stripped == "---":
            continue
        if "Carris GTFS-RT" in stripped:
            freshness_note = (
                "dados em tempo real da Carris disponíveis."
                if language == "pt"
                else "Carris real-time data available."
            )
            continue
        if "Real-Time Data Active" in stripped:
            freshness_note = (
                "feed de veículos Carris ativo."
                if language == "pt"
                else "Carris real-time vehicle feed active."
            )
            continue
        group_match = re.search(r"\*\*\[(?P<route>[^\]]+)\]\s+Para\s+(?P<dest>.+?)\*\*", stripped)
        if group_match:
            current_route = group_match.group("route").strip()
            current_destination = group_match.group("dest").strip()
            continue
        times_match = re.search(r"(?:🕒|ðŸ•’)\s*(?P<times>.+)", stripped)
        if times_match and current_route:
            groups.append((current_route, current_destination, times_match.group("times").strip()))
            current_route = ""
            current_destination = ""

    if not groups:
        return raw

    first_route, first_destination, first_times = groups[0]
    first_time = re.split(r",|\s+\(", first_times)[0].strip("* ")

    if language == "pt":
        vehicle_name = "elétrico" if icon == "🚋" else "autocarro"
        title_line = (
            f"### {icon} **Próximas partidas do {line_label or first_route} em {stop_label}**"
        )
        direct_line = (
            f"✅ **Resposta direta:** o próximo {vehicle_name} **{line_label or first_route}** "
            f"para **{first_destination}** está listado para **{first_time}**."
        )
        lines = [title_line, "", direct_line, "", "---", ""]
        if freshness_note:
            lines.append(f"- 📡 **Tempo real:** {freshness_note}")
        for route, destination, times in groups:
            localized_times = re.sub(r"\(\+(\d+)\s+more\)", r"(+\1 restantes)", times)
            lines.append(f"- {icon} **{route} → {destination}:** {localized_times}")
        return "\n".join(lines).strip()

    vehicle_name = "tram" if icon == "🚋" else "bus"
    title_line = f"### {icon} **Next {line_label or first_route} departures at {stop_label}**"
    direct_line = (
        f"✅ **Direct answer:** the next {vehicle_name} **{line_label or first_route}** "
        f"towards **{first_destination}** is listed for **{first_time}**."
    )
    lines = [title_line, "", direct_line, "", "---", ""]
    if freshness_note:
        lines.append(f"- 📡 **Real time:** {freshness_note}")
    for route, destination, times in groups:
        lines.append(f"- {icon} **{route} → {destination}:** {times}")
    return "\n".join(lines).strip()


def _build_deterministic_carris_stop_response(user_message: str, language: str = "pt") -> Optional[str]:
    """Builds deterministic Carris stop/line wait responses from the best matching tool."""
    request = _parse_carris_line_stop_query(user_message)
    if not request:
        return None

    from tools.carris_api import (
        carris_get_arrivals,
        carris_get_next_departures,
        carris_vehicle_eta,
    )

    kind = request.get("kind")
    line = request.get("line")
    stop_id = request.get("stop_id")
    stop_name = request.get("stop_name")

    if not stop_id and stop_name:
        stop_id, resolved_stop_name = _resolve_carris_stop(stop_name)
        stop_name = resolved_stop_name or stop_name

    if kind == "arrivals" and stop_id:
        return str(carris_get_arrivals.invoke({"stop_id": stop_id, "limit": 8})).strip()

    if kind == "eta" and line and stop_name:
        return str(
            carris_vehicle_eta.invoke(
                {"route_short_name": line, "stop_name": stop_name}
            )
        ).strip()

    if kind == "departures" and line and stop_id:
        raw_departures = str(
            carris_get_next_departures.invoke(
                {"stop_id": stop_id, "route_short_name": line, "limit": 8}
            )
        ).strip()
        return _format_carris_next_departures_output(
            raw_departures,
            line=line,
            stop_name=stop_name or stop_id,
            language=language,
        )

    return None


def _extract_cp_route_name(query: str) -> Optional[str]:
    """Extracts a canonical CP route name from a natural-language query."""
    normalized = _normalize_token(query)
    alias_map = {
        "linha de sintra": "Sintra",
        "sintra line": "Sintra",
        "sintra": "Sintra",
        "linha de cascais": "Cascais",
        "cascais line": "Cascais",
        "cascais": "Cascais",
        "linha da azambuja": "Azambuja",
        "azambuja line": "Azambuja",
        "azambuja": "Azambuja",
        "linha do sado": "Sado",
        "sado line": "Sado",
        "sado": "Sado",
    }

    for alias, canonical in alias_map.items():
        if alias in normalized:
            return canonical
    return None


def _is_probable_cp_suburban_endpoint(fragment: str) -> bool:
    """Return whether a route endpoint is a known CP suburban/AML anchor."""
    normalized = _normalize_token(fragment)
    return any(anchor in normalized for anchor in _CP_AML_ROUTE_ANCHORS)


def _is_probable_cp_suburban_route_pair(endpoints: Tuple[str, str]) -> bool:
    """Return whether both endpoints are likely served by CP suburban/AML data."""
    return all(_is_probable_cp_suburban_endpoint(endpoint) for endpoint in endpoints)


def _infer_cp_line_label_for_pair(origin: str, destination: str, language: str = "pt") -> str:
    """Infer a shared CP suburban line label for a known station pair."""
    try:
        from tools.cp_api import CP_LINES, get_cp_station_info
    except Exception:
        return ""

    origin_info = get_cp_station_info(origin)
    destination_info = get_cp_station_info(destination)
    if not origin_info or not destination_info:
        return ""

    common_lines = [
        line
        for line in (origin_info.get("lines") or [])
        if line in set(destination_info.get("lines") or [])
    ]
    if not common_lines:
        return ""

    labels: list[str] = []
    for line_key in common_lines:
        line_info = CP_LINES.get(str(line_key).lower(), {})
        label = str(line_info.get("name") or line_key).strip()
        if label:
            labels.append(label)
    return ", ".join(dict.fromkeys(labels))


def _build_cp_tool_spec(user_message: str) -> Optional[Dict[str, Any]]:
    """Maps common natural-language CP queries to deterministic tool specs."""
    query = user_message.strip()
    query_lower = query.lower()
    route_name = _extract_cp_route_name(query)
    endpoints = _extract_route_endpoints(query)
    cp_route_pair = bool(endpoints and _is_probable_cp_suburban_route_pair(endpoints))
    train_exclusion_context = bool(
        re.search(
            r"\b(?:evitar|evito|sem|dispensar|prefiro\s+evitar|n[aã]o\s+quero|avoid|without|no)\s+(?:o\s+|a\s+|the\s+)?(?:cp|comboio|comboios|combios|train|trains)\b",
            query_lower,
        )
    )
    explicit_train_context = bool(
        re.search(r"\b(cp|comboio|comboios|combios|train|trains)\b", query_lower)
        and not train_exclusion_context
    )
    explicit_metro_context = bool(
        re.search(r"\b(?:metro|metropolitano|linha\s+(?:azul|verde|amarela|vermelha))\b", query_lower)
    )
    explicit_carris_urban_context = bool(
        re.search(r"\b(carris|autocarro|autocarros|bus|buses|tram|trams|el[eé]trico|eletrico)\b", query_lower)
    )
    if explicit_metro_context and not explicit_train_context:
        return None
    if explicit_carris_urban_context and train_exclusion_context:
        return None
    if explicit_carris_urban_context and not explicit_train_context:
        return None
    if endpoints and cp_route_pair and _is_generic_public_transport_route_query(query) and not explicit_train_context:
        cp_route_pair = False
    has_train_context = bool(
        explicit_train_context
        or cp_route_pair
        or (
            route_name
            and re.search(
                r"\b(frequency|headway|how often|frequ[eê]ncia|de quanto em quanto tempo)\b",
                query_lower,
            )
        )
    )
    if not has_train_context:
        return None

    schedule_patterns = [
        r"(?:next|upcoming)\s+(?:train\s+)?departures\s+(?:from|at)\s+(?P<station>.+?)(?:[\?\!\.,;]|$)",
        r"(?:next|upcoming)\s+trains\s+(?:from|at)\s+(?P<station>.+?)(?:[\?\!\.,;]|$)",
        r"pr[oó]xim(?:os|as)\s+comboios\s+(?:de|em)\s+(?P<station>.+?)(?:[\?\!\.,;]|$)",
        r"hor[aá]rios?\s+dos?\s+comboios\s+(?:de|em)\s+(?P<station>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in schedule_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {
                "name": "get_train_schedule",
                "args": {"station_name": _resolve_cp_station_name(match.group("station"))},
            }

    if route_name and re.search(
        r"\b(frequency|headway|how often|frequ[eê]ncia|de quanto em quanto tempo)\b",
        query_lower,
    ):
        return {"name": "get_train_frequency", "args": {"route_name": route_name}}

    if endpoints and explicit_train_context:
        return {
            "name": "plan_train_trip",
            "args": {
                "origin": _resolve_cp_station_name(endpoints[0]),
                "destination": _resolve_cp_station_name(endpoints[1]),
            },
        }

    if endpoints and cp_route_pair:
        return {
            "name": "plan_train_trip",
            "args": {
                "origin": _resolve_cp_station_name(endpoints[0]),
                "destination": _resolve_cp_station_name(endpoints[1]),
            },
        }

    if re.search(r"\b(status|delay|delays|running|atrasos?|a funcionar)\b", query_lower):
        return {"name": "get_train_status", "args": {}}

    if re.search(
        r"(?:\b(cp|train|comboio)s?\b.*\b(routes|lines|linhas)\b)|(?:\b(routes|lines|linhas)\b.*\b(cp|train|comboio)s?\b)",
        query_lower,
    ):
        return {"name": "get_cp_routes", "args": {}}

    station_search_patterns = [
        r"(?:cp|train)\s+stations?\s+(?:for|near|named)\s+(?P<term>.+?)(?:[\?\!\.,;]|$)",
        r"(?:which|what)\s+train\s+stations?\s+(?:are\s+)?(?:closest\s+to|near|around)\s+(?P<term>.+?)(?:[\?\!\.,;]|$)",
        r"esta[cç][õo]es?\s+cp\s+(?:para|de)\s+(?P<term>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in station_search_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {
                "name": "search_cp_stations",
                "args": {"query": _resolve_cp_station_name(match.group("term"))},
            }

    if route_name == "Cascais" and re.fullmatch(r"(?:cp|comboio|comboios|combios|train|trains)\s+cascais", query_lower):
        return {
            "name": "plan_train_trip",
            "args": {
                "origin": "Cais do Sodré",
                "destination": "Cascais",
            },
        }

    return None


def _looks_like_carris_metropolitana_query(
    user_message: str,
    endpoints: Optional[Tuple[str, str]] = None,
) -> bool:
    """Heuristically identifies suburban-bus queries worth routing to Carris Metropolitana."""
    normalized = _normalize_token(user_message)
    if "carris metropolitana" in normalized:
        return True

    suburban_hints = {
        "almada",
        "cacilhas",
        "oeiras",
        "amadora",
        "sintra",
        "loures",
        "montijo",
        "seixal",
        "barreiro",
        "moita",
        "alcochete",
        "palmela",
        "sesimbra",
        "setubal",
        "almada forum",
    }

    if any(hint in normalized for hint in suburban_hints):
        return True

    if endpoints:
        endpoint_tokens = [_normalize_token(part) for part in endpoints]
        return any(hint in token for token in endpoint_tokens for hint in suburban_hints)

    # Carris Metropolitana public line identifiers are four-digit codes in
    # the 1xxx-4xxx families. Carris Urban uses shorter codes such as 15E,
    # 7xx, or 2xx, so a live/location question for "autocarro 3701" should
    # not fall through to Carris Urban just because the operator name is
    # omitted.
    if re.search(r"\b[1-4]\d{3}[a-z]?\b", normalized) and re.search(
        r"\b(?:autocarro|autocarros|bus|buses|linha|line|onde|where|agora|now|gps|location|position)\b",
        normalized,
    ):
        return True

    return False


def _extract_aml_municipality_area_hint(query: str) -> str:
    """Extract a canonical AML municipality name from a free-form query."""
    candidates = [
        *sorted(AML_MUNICIPALITY_NAMES, key=len, reverse=True),
        "Setubal",
        "Vila Franca",
    ]
    for known_area in candidates:
        if re.search(rf"\b{re.escape(known_area)}\b", query, flags=re.IGNORECASE):
            if known_area == "Setubal":
                return "Setúbal"
            if known_area == "Vila Franca":
                return "Vila Franca de Xira"
            return known_area
    return ""


def _build_carris_metropolitana_tool_spec(user_message: str) -> Optional[Dict[str, Any]]:
    """Maps common Carris Metropolitana user requests to deterministic tool specs."""
    query = user_message.strip()
    query_lower = query.lower()
    endpoints = _extract_route_endpoints(query)
    has_cm_context = _looks_like_carris_metropolitana_query(query, endpoints=endpoints)
    if not has_cm_context:
        return None

    if endpoints and (
        re.search(r"\b(direct|diret[ao]s?)\b", query_lower)
        and re.search(r"\b(disruptions?|alerts?|alertas?)\b", query_lower)
        and re.search(r"\b(where|location|position|now|agora|momento)\b", query_lower)
    ):
        return None

    if re.search(r"\b(alerts?|alertas?|disruptions?)\b", query_lower):
        line_match = re.search(r"\b(?:line|linha)\s+(?P<line_id>\d{3,4}[a-z]?)\b", query_lower)
        area_match = re.search(
            r"\b(?:around|near|in|for|para|em|na|no|perto de|junto de)\s+(?P<area>.+?)(?:\s+today|\s+hoje|[\?\!\.,;]|$)",
            query,
            flags=re.IGNORECASE,
        )
        area_fallback = _extract_aml_municipality_area_hint(query)
        args: Dict[str, Any] = {}
        if line_match:
            args["line"] = line_match.group("line_id").upper()
        elif area_fallback:
            args["area"] = area_fallback
        elif area_match:
            area_candidate = _clean_query_fragment(area_match.group("area"))
            if _normalize_token(area_candidate) not in {"carris metropolitana", "metropolitana"}:
                args["area"] = area_candidate
        return {"name": "get_carris_metropolitana_alerts", "args": args}

    stop_id_match = re.search(r"\b(?:stop id|stop|paragem)\s+(?P<stop_id>\d{3,})\b", query_lower)
    line_id_match = re.search(r"\b(?:line|linha)\s+(?P<line_id>\d{3,4}[a-z]?)\b", query_lower)
    if not line_id_match and re.search(
        r"\b(waiting for|waiting|esperar|esperando|bus|buses|autocarro|autocarros)\b",
        query_lower,
    ):
        line_id_match = re.search(r"\b(?P<line_id>\d{3,4}[a-z]?)\b", query_lower)

    if stop_id_match and re.search(r"\b(info|information|details|detalhes?)\b", query_lower):
        return {
            "name": "get_carris_metropolitana_stop_info",
            "args": {"stop_id": stop_id_match.group("stop_id")},
        }

    if stop_id_match and line_id_match and re.search(
        r"\b(next|departures|pr[oó]ximas?|partidas?)\b",
        query_lower,
    ):
        return {
            "name": "get_bus_next_departures",
            "args": {
                "line_id": line_id_match.group("line_id").upper(),
                "stop_id": stop_id_match.group("stop_id"),
            },
        }

    if line_id_match and re.search(
        r"\b(where|gps|location|locations|position|positions|onde|localiza[cç][aã]o|localiza[cç][õo]es)\b",
        query_lower,
    ):
        location_match = re.search(
            r"\b(?:in|at|near|around|em|na|no|perto de)\s+(?P<location>[A-Za-zÀ-ÿ\s-]+?)\s*(?:right now|now|agora|[\?\!\.,;]|$)",
            query,
            flags=re.IGNORECASE,
        )
        args = {"line_id": line_id_match.group("line_id").upper()}
        if location_match:
            location = _clean_query_fragment(location_match.group("location"))
            if not re.fullmatch(r"(?:the\s+)?line", location, flags=re.IGNORECASE):
                args["location"] = location
        return {
            "name": "get_bus_realtime_locations",
            "args": args,
        }

    near_match = re.search(
        r"(?:near|perto de|around)\s+(?P<location>.+?)(?:[\?\!\.,;]|$)",
        query,
        flags=re.IGNORECASE,
    )
    if not near_match and re.search(r"\b(nearby|por perto)\b", query_lower):
        near_match = re.search(
            r"(?:i(?:'m| am)|estou)\s+(?:at|in|em|na|no)\s+(?P<location>.+?)(?:[\?\!\.,;]|$)",
            query,
            flags=re.IGNORECASE,
        )
    if near_match and re.search(r"\b(bus|buses|autocarro|autocarros)\b", query_lower):
        location_fragment = re.sub(
            r"\b(right now|agora)\b",
            "",
            near_match.group("location"),
            flags=re.IGNORECASE,
        )
        return {
            "name": "get_real_time_bus_positions",
            "args": {"location": _clean_query_fragment(location_fragment), "radius_km": 1.0},
        }

    if endpoints and re.search(r"\b(bus|buses|autocarro|autocarros)\b", query_lower):
        if re.search(r"\b(direct|diret[ao]s?)\b", query_lower):
            return {
                "name": "find_direct_bus_lines",
                "args": {"origin": endpoints[0], "destination": endpoints[1]},
            }
        return {
            "name": "find_bus_routes",
            "args": {"origin": endpoints[0], "destination": endpoints[1]},
        }

    if endpoints and re.search(
        r"\b(transportes?\s+p[uú]blicos?|transportes?|public\s+transport|transport|transporte|rota|route|"
        r"como\s+(?:vou|chego)|how\s+(?:do\s+i\s+)?(?:get|go)|quero\s+ir|want\s+to\s+go)\b",
        query_lower,
    ):
        return {
            "name": "find_bus_routes",
            "args": {"origin": endpoints[0], "destination": endpoints[1]},
        }

    line_search_patterns = [
        r"carris metropolitana\s+lines?\s+(?:for|to|near)\s+(?P<term>.+?)(?:[\?\!\.,;]|$)",
        r"(?:which|what)\s+(?:bus\s+)?lines?\s+(?:serve|servem|go to|run to|go through|run through)\s+(?P<term>.+?)(?:[\?\!\.,;]|$)",
        r"linhas?\s+da\s+carris\s+metropolitana\s+(?:para|de)\s+(?P<term>.+?)(?:[\?\!\.,;]|$)",
    ]
    for pattern in line_search_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {
                "name": "search_carris_metropolitana_lines",
                "args": {"query": _clean_query_fragment(match.group("term"))},
            }

    return None


def _metropolitana_line_serves_location(line_id: str, location: str) -> Tuple[bool, str]:
    """Check whether a Carris Metropolitana line metadata mentions a requested area."""
    if not line_id or not location:
        return True, ""
    try:
        from tools.carrismetropolitana_api import load_carris_metropolitana_lines
    except Exception:
        return True, ""

    normalized_location = _normalize_token(location)
    normalized_line_id = str(line_id).strip().upper()
    for line in load_carris_metropolitana_lines():
        identifiers = {
            str(line.get("id") or "").upper(),
            str(line.get("short_name") or "").upper(),
        }
        if normalized_line_id not in identifiers:
            continue
        searchable_parts = [
            str(line.get("long_name") or ""),
            *[str(item) for item in line.get("localities") or []],
            *[str(item) for item in line.get("municipalities") or []],
        ]
        searchable = _normalize_token(" ".join(searchable_parts))
        return normalized_location in searchable, str(line.get("long_name") or "")
    return True, ""


def _extract_first_metropolitana_line_id(text: str) -> Optional[str]:
    """Extract the first public-facing Carris Metropolitana line ID from tool text."""
    match = re.search(r"\b(?:Line|Linha)\s+(?P<line>\d{3,4}[A-Z]?)\b", text or "", flags=re.IGNORECASE)
    if match:
        return match.group("line").upper()
    match = re.search(r"\b(?P<line>[1-4]\d{3}[A-Z]?)\b", text or "")
    return match.group("line").upper() if match else None


def _summarize_relevant_alerts_for_line(alert_text: str, line_id: Optional[str]) -> str:
    """Summarize whether alert output appears to mention a recommended line."""
    if not alert_text:
        return "- ℹ️ **Service disruptions:** not confirmed from the alert feed."
    if "No active" in alert_text or "No active Carris" in alert_text:
        return "- ✅ **Service disruptions:** no active alert found for the requested area."
    if line_id and re.search(rf"\b{re.escape(line_id)}\b", alert_text):
        return f"- ⚠️ **Service disruptions:** at least one current alert mentions line {line_id}; check the alert details before travelling."
    return "- ℹ️ **Service disruptions:** active area alerts exist, but none were clearly tied to the recommended line in the returned summary."


def _build_deterministic_metro_route_response(
    user_message: str,
    context: str,
) -> Optional[str]:
    """Builds a deterministic metro route answer directly from tool outputs."""
    endpoints = _extract_route_endpoints(user_message)
    if not endpoints:
        return None

    language = _infer_language(user_message, context)
    future_planning = _is_future_transport_planning_query(user_message)
    preferences = _parse_route_mode_preferences(user_message)

    from tools.metrolisboa_api import (
        METRO_LINES,
        get_landmark_info,
        get_metro_live_service_evidence,
        get_metro_regular_service_context,
    )
    from tools.transport_api import get_route_between_stations

    try:
        route_result = str(
            get_route_between_stations.invoke(
                {"origin": endpoints[0], "destination": endpoints[1]}
            )
        )
    except Exception:
        return None

    if not _route_result_uses_metro(route_result):
        return None
    if _route_result_is_metro_only_partial(route_result) and not preferences["metro_only"]:
        return None

    details = _parse_route_details(route_result)
    board_station = details["board_station"] or endpoints[0].title()
    final_station = details["final_station"]
    transfer_station = details["transfer_station"]
    directions = details["directions"]
    first_direction = directions[0] if directions else None
    second_direction = directions[1] if len(directions) > 1 else None
    estimated_time = details["estimated_time"]
    walk_target = details["walk_target"]
    origin_display = _get_transport_display_name(endpoints[0])
    destination_display = _get_transport_display_name(endpoints[1])
    destination_detailed_display = _get_transport_display_name(endpoints[1], detailed=True)
    walk_target_display = _get_transport_display_name(walk_target) if walk_target else None
    board_station_display = _display_metro_line_station_name(board_station)
    final_station_display = _display_metro_line_station_name(final_station)
    transfer_station_display = _display_metro_line_station_name(transfer_station) if transfer_station else None
    board_station_route_display = _display_metro_station_with_line_badges(board_station, language)
    final_station_route_display = _display_metro_station_with_line_badges(final_station, language)
    transfer_station_route_display = (
        _display_metro_station_with_line_badges(transfer_station, language) if transfer_station else None
    )
    first_direction_display = _display_metro_line_station_name(first_direction) if first_direction else None
    second_direction_display = _display_metro_line_station_name(second_direction) if second_direction else None
    destination_landmark = get_landmark_info(endpoints[1]) or (
        get_landmark_info(walk_target) if walk_target else None
    )

    if not final_station:
        return None

    metro_service_context = get_metro_regular_service_context(language=language)
    metro_regular_service_open = bool(
        metro_service_context["is_regular_service_open"]
    )
    metro_live_evidence: Dict[str, Any] = {}
    if not future_planning and not metro_regular_service_open:
        metro_live_evidence = get_metro_live_service_evidence()

    first_line_id = _get_line_id_between(board_station, transfer_station or final_station)
    second_line_id = _get_line_id_between(transfer_station, final_station) if transfer_station else None
    line_ids = [line_id for line_id in [first_line_id, second_line_id] if line_id]

    state_lines = [] if future_planning else _build_route_state_lines(line_ids, language)

    if not future_planning and _route_has_interrupted_green_segment(
        first_line_id,
        second_line_id,
        board_station,
        transfer_station,
        final_station,
    ):
        safer_route = _build_disruption_safe_metro_route(
            endpoints[0],
            endpoints[1],
            language,
            board_station=board_station,
            final_station=final_station,
        )
        if safer_route:
            return safer_route

    wait_targets: List[Tuple[str, str]] = []
    if board_station and first_direction:
        wait_targets.append((board_station, first_direction))
    if transfer_station and second_direction:
        wait_targets.append((transfer_station, second_direction))
    if not wait_targets:
        wait_targets = _parse_wait_targets_from_route(route_result)

    realtime_lines = [] if future_planning else _build_metro_wait_lines(wait_targets, language)

    tip = _build_practical_tip(
        language=language,
        first_direction=first_direction_display or first_direction,
        transfer_station=transfer_station_display or transfer_station,
        second_line_id=second_line_id,
        final_station=final_station_display,
        walk_target=walk_target,
        walk_target_display=walk_target_display,
        destination_display=destination_detailed_display,
        destination_landmark=destination_landmark,
    )
    additional_options = _build_additional_route_options(
        user_message=user_message,
        origin=endpoints[0],
        destination=endpoints[1],
        language=language,
    )

    route_title = f"### 🚇 **{origin_display} → {destination_display}**"
    state_title = _metro_state_title(state_lines, language)
    time_title = "⏳ **Tempo total estimado:**" if language == "pt" else "⏳ **Estimated total time:**"
    route_section = "🗺️ **O seu Trajeto de Metro:**" if language == "pt" else "🗺️ **Your Metro Route:**"
    waits_section = "🗓️ **Próximos Metros** (tempo real):" if language == "pt" else "🗓️ **Next Metros** (real time):"
    tip_title = "💡 **Dica rápida:**" if language == "pt" else "💡 **Quick tip:**"
    source_label = "📌 **Fonte:**" if language == "pt" else "📌 **Source:**"
    updated_label = "**Atualizado:**" if language == "pt" else "**Updated:**"
    board_text = "- 📍 **Embarque na estação" if language == "pt" else "- 📍 **Board at"
    exit_text = "- 🎯 **Saia na estação" if language == "pt" else "- 🎯 **Exit at"
    transfer_text = "- 🔄 **Transferência em" if language == "pt" else "- 🔄 **Transfer at"
    walk_text = "- 🚶 **Siga a pé para" if language == "pt" else "- 🚶 **Walk to"
    direction_word = "direção" if language == "pt" else "direction"

    response_lines = [route_title, ""]
    if language == "pt":
        if transfer_station:
            if not future_planning and not metro_regular_service_open:
                direct_answer = (
                    f"✅ **Resposta direta:** a rota de Metro é de "
                    f"**{board_station_display}** para **{final_station_display}**, "
                    f"com transbordo em **{transfer_station_display or transfer_station}**; "
                    "neste momento está fora do horário regular, por isso confirma "
                    "operação especial antes de sair."
                )
            else:
                direct_answer = (
                    f"✅ **Resposta direta:** vai de metro de "
                    f"**{board_station_display}** para **{final_station_display}**, "
                    f"com transbordo em **{transfer_station_display or transfer_station}**."
                )
            response_lines.extend(
                [
                    direct_answer,
                    "",
                    "---",
                    "",
                ]
            )
        elif first_line_id:
            if not future_planning and not metro_regular_service_open:
                direct_answer = (
                    f"✅ **Resposta direta:** a rota de Metro usa a "
                    f"**{_line_display_name(first_line_id, language)}** de "
                    f"**{board_station_display}** para **{final_station_display}**; "
                    "neste momento está fora do horário regular, por isso confirma "
                    "operação especial antes de sair."
                )
            else:
                direct_answer = (
                    f"✅ **Resposta direta:** usa a "
                    f"**{_line_display_name(first_line_id, language)}** diretamente "
                    f"de **{board_station_display}** para **{final_station_display}**."
                )
            response_lines.extend(
                [
                    direct_answer,
                    "",
                    "---",
                    "",
                ]
            )
    else:
        if transfer_station:
            if not future_planning and not metro_regular_service_open:
                direct_answer = (
                    f"✅ **Direct answer:** the Metro route is from "
                    f"**{board_station_display}** to **{final_station_display}**, "
                    f"transferring at **{transfer_station_display or transfer_station}**; "
                    "right now Metro is outside regular hours, so confirm special "
                    "service before leaving."
                )
            else:
                direct_answer = (
                    f"✅ **Direct answer:** take the Metro from "
                    f"**{board_station_display}** to **{final_station_display}**, "
                    f"transferring at **{transfer_station_display or transfer_station}**."
                )
            response_lines.extend(
                [
                    direct_answer,
                    "",
                    "---",
                    "",
                ]
            )
        elif first_line_id:
            if not future_planning and not metro_regular_service_open:
                direct_answer = (
                    f"✅ **Direct answer:** the Metro route uses the "
                    f"**{_line_display_name(first_line_id, language)}** from "
                    f"**{board_station_display}** to **{final_station_display}**; "
                    "right now Metro is outside regular hours, so confirm special "
                    "service before leaving."
                )
            else:
                direct_answer = (
                    f"✅ **Direct answer:** take the "
                    f"**{_line_display_name(first_line_id, language)}** directly "
                    f"from **{board_station_display}** to **{final_station_display}**."
                )
            response_lines.extend(
                [
                    direct_answer,
                    "",
                    "---",
                    "",
                ]
            )

    if state_lines:
        response_lines.extend([
            state_title,
            *state_lines,
            "",
        ])

    if estimated_time:
        response_lines.extend([f"{time_title} {estimated_time}", ""])

    response_lines.extend([
        route_section,
        f"{board_text} {board_station_route_display}**",
    ])

    if first_line_id and first_direction:
        response_lines.append(
            f"- {METRO_LINES[first_line_id]['emoji']} **{_line_display_name(first_line_id, language)}** - {direction_word} **{first_direction_display or first_direction}**"
        )

    if transfer_station:
        response_lines.append(f"{transfer_text} {transfer_station_route_display or transfer_station_display or transfer_station}**")
        if second_line_id and second_direction:
            response_lines.append(
                f"- {METRO_LINES[second_line_id]['emoji']} **{_line_display_name(second_line_id, language)}** - {direction_word} **{second_direction_display or second_direction}**"
            )

    response_lines.append(f"{exit_text} {final_station_route_display}**")

    if walk_target and _normalize_token(walk_target) != _normalize_token(final_station):
        response_lines.append(f"{walk_text} {walk_target_display or walk_target}**")

    if future_planning:
        response_lines.extend([
            "",
            _build_future_realtime_note(language),
            "",
        ])
    else:
        if not metro_regular_service_open:
            if language == "pt":
                realtime_lines = [
                    "- 🌙 Não apresento próximos metros como serviço ativo porque "
                    "o Metro está fora do horário regular.",
                ]
                if metro_live_evidence.get("has_live_wait_times"):
                    realtime_lines.append(
                        "- 📡 Há tempos de espera em tempo real em "
                        f"{metro_live_evidence.get('station')}; pode existir "
                        "operação especial, mas confirma no operador."
                    )
                else:
                    realtime_lines.append(
                        "- 📡 Se houver operação especial, confirma os tempos de "
                        "espera no Metro de Lisboa antes de sair."
                    )
            else:
                realtime_lines = [
                    "- 🌙 I am not presenting next trains as active service because "
                    "Metro is outside regular operating hours.",
                ]
                if metro_live_evidence.get("has_live_wait_times"):
                    realtime_lines.append(
                        "- 📡 Live waits are available at "
                        f"{metro_live_evidence.get('station')}; special service may "
                        "be active, but confirm with the operator."
                    )
                else:
                    realtime_lines.append(
                        "- 📡 If special service is running, confirm live waits with "
                        "Metro de Lisboa before leaving."
                    )
        response_lines.extend([
            "",
            waits_section,
            *realtime_lines,
            "",
        ])

    if tip:
        response_lines.append(f"{tip_title} {tip}")
        response_lines.append("")

    if additional_options:
        response_lines.extend(additional_options)

    response_lines.append(
        f"{source_label} [*Metro de Lisboa*](https://www.metrolisboa.pt) | {updated_label} {datetime.now().strftime('%H:%M')}"
    )

    return "\n".join(response_lines).strip()


def _build_deterministic_route_tool_response(user_message: str) -> Optional[str]:
    """Returns the raw route-tool guidance for non-metro-direct routes."""
    if _query_requests_metro_line_wait_times(user_message):
        return None

    raw_endpoints = _extract_route_endpoints(user_message, collapse_generic_service_area=False)
    endpoints = _extract_route_endpoints(user_message)
    if not endpoints:
        return None
    language = _infer_language(user_message, "")
    raw_destination = raw_endpoints[1] if raw_endpoints else endpoints[1]
    area_destination = _generic_service_area_endpoint(raw_destination)

    if _query_has_wait_departure_intent(user_message) and re.search(
        r"\b(carris|autocarro|autocarros|bus|buses|tram|trams|el[eé]trico|eletrico)\b",
        user_message,
        flags=re.IGNORECASE,
    ) and not endpoints:
        return None

    bus_or_tram_requested = bool(
        re.search(
            r"\b(carris|autocarro|autocarros|bus|buses|tram|trams|el[eé]trico|eletrico)\b",
            user_message,
            flags=re.IGNORECASE,
        )
    )

    if bus_or_tram_requested:
        try:
            from tools.carris_api import carris_find_routes_between

            carris_result = str(
                carris_find_routes_between.invoke(
                    {"origin": endpoints[0], "destination": endpoints[1], "search_radius_km": 0.8}
                )
            ).strip()
        except Exception:
            carris_result = ""

        if carris_result and not _tool_result_indicates_no_match(carris_result):
            response = _build_carris_surface_route_response(
                carris_result,
                user_message,
                endpoints[0],
                endpoints[1],
            )
            return _append_generic_service_area_note(response, raw_destination, area_destination, language)

    from tools.transport_api import get_route_between_stations

    try:
        route_result = str(
            get_route_between_stations.invoke(
                {"origin": endpoints[0], "destination": endpoints[1]}
            )
        )
    except Exception:
        return None

    route_result = route_result.strip() if route_result else ""
    if not route_result:
        return None

    if _normalize_token(endpoints[0]) == _normalize_token(endpoints[1]):
        return route_result

    is_metro_route = _route_result_uses_metro(route_result)
    fastest_requested = bool(
        re.search(r"\b(fastest|quickest|mais r[aá]pid[ao]|mais depressa)\b", user_message, flags=re.IGNORECASE)
    )
    if is_metro_route and _route_result_is_metro_only_partial(route_result):
        broad_destination = _bare_aml_municipality_label(endpoints[1])
        cp_first = bool(broad_destination and not bus_or_tram_requested)

        if cp_first:
            cp_bridge = _build_cp_bridge_for_partial_metro_route(
                user_message=user_message,
                origin=endpoints[0],
                destination=endpoints[1],
                route_result=route_result,
                checked_metropolitana_direct=True,
            )
            if cp_bridge:
                return cp_bridge

        if _is_generic_public_transport_route_query(user_message) and not fastest_requested and not (is_metro_route and not _route_result_is_metro_only_partial(route_result)):
            try:
                from tools.carris_api import carris_find_routes_between

                route_preferences = _parse_route_mode_preferences(user_message)
                broad_surface_search = bool(
                    route_preferences.get("alternative_mode_request")
                    or re.search(
                        r"\b(?:alternativas?|op[cç][oõ]es|options|sem\s+metro|without\s+metro)\b",
                        user_message,
                        flags=re.IGNORECASE,
                    )
                )
                urban_carris_result = str(
                    carris_find_routes_between.invoke(
                        {
                            "origin": endpoints[0],
                            "destination": endpoints[1],
                            "search_radius_km": 0.8 if broad_surface_search else 0.4,
                        }
                    )
                ).strip()
            except Exception:
                urban_carris_result = ""
            if (
                (not urban_carris_result or _tool_result_indicates_no_match(urban_carris_result))
                and not broad_surface_search
            ):
                try:
                    urban_carris_result = str(
                        carris_find_routes_between.invoke(
                            {
                                "origin": endpoints[0],
                                "destination": endpoints[1],
                                "search_radius_km": 0.8,
                            }
                        )
                    ).strip()
                except Exception:
                    urban_carris_result = ""
            if urban_carris_result and not _tool_result_indicates_no_match(urban_carris_result):
                response = _build_carris_surface_route_response(
                    urban_carris_result,
                    user_message,
                    endpoints[0],
                    endpoints[1],
                )
                return _append_generic_service_area_note(response, raw_destination, area_destination, language)

        metropolitana_bridge = _build_metropolitana_bridge_for_partial_metro_route(
            user_message=user_message,
            origin=endpoints[0],
            destination=endpoints[1],
            route_result=route_result,
        )
        if metropolitana_bridge:
            return metropolitana_bridge

        if not cp_first:
            cp_bridge = _build_cp_bridge_for_partial_metro_route(
                user_message=user_message,
                origin=endpoints[0],
                destination=endpoints[1],
                route_result=route_result,
                checked_metropolitana_direct=True,
            )
            if cp_bridge:
                return cp_bridge

        if bus_or_tram_requested:
            if broad_destination:
                return _build_bare_municipality_clarification(
                    origin=endpoints[0],
                    destination=endpoints[1],
                    municipality=broad_destination,
                    language=_infer_language(user_message, ""),
                )
            return _build_mode_unavailable_response(
                mode="autocarro" if _infer_language(user_message, "") == "pt" else "bus",
                origin=endpoints[0],
                destination=endpoints[1],
                language=_infer_language(user_message, ""),
            )

    if fastest_requested and is_metro_route and _is_generic_public_transport_route_query(user_message):
        return route_result

    if is_metro_route and not _is_generic_public_transport_route_query(user_message):
        return route_result

    route_preferences = _parse_route_mode_preferences(user_message)
    try:
        from tools.carris_api import carris_find_routes_between

        broad_surface_search = bool(
            route_preferences.get("alternative_mode_request")
            or re.search(
                r"\b(?:alternativas?|op[cç][oõ]es|options|sem\s+metro|without\s+metro)\b",
                user_message,
                flags=re.IGNORECASE,
            )
        )
        carris_result = str(
            carris_find_routes_between.invoke(
                {
                    "origin": endpoints[0],
                    "destination": endpoints[1],
                    "search_radius_km": 0.8 if broad_surface_search else 0.4,
                }
            )
        ).strip()
    except Exception:
        broad_surface_search = False
        carris_result = ""

    valid_carris_result = carris_result and not any(
        marker in carris_result
        for marker in [
            "No direct Carris route found",
            "Could not locate",
            "Sem rota direta",
            "Não foi possível localizar",
        ]
    )

    if (
        not valid_carris_result
        and _is_generic_public_transport_route_query(user_message)
        and not broad_surface_search
    ):
        try:
            from tools.carris_api import carris_find_routes_between

            carris_result = str(
                carris_find_routes_between.invoke(
                    {
                        "origin": endpoints[0],
                        "destination": endpoints[1],
                        "search_radius_km": 0.8,
                    }
                )
            ).strip()
        except Exception:
            carris_result = ""
        valid_carris_result = carris_result and not any(
            marker in carris_result
            for marker in [
                "No direct Carris route found",
                "Could not locate",
                "Sem rota direta",
                "NÃ£o foi possÃ­vel localizar",
            ]
        )

    comparison_requested = bool(route_preferences.get("alternative_mode_request"))
    if valid_carris_result and _is_generic_public_transport_route_query(user_message) and is_metro_route:
        formatted_carris_response = _build_carris_surface_route_response(
            carris_result,
            user_message,
            endpoints[0],
            endpoints[1],
        )
        if not fastest_requested and not comparison_requested and not (is_metro_route and not _route_result_is_metro_only_partial(route_result)):
            return _append_generic_service_area_note(formatted_carris_response, raw_destination, area_destination, language)

        updated_label = "Atualizado" if _infer_language(user_message, "") == "pt" else "Updated"
        source_label = "Fonte" if _infer_language(user_message, "") == "pt" else "Source"
        title = (
            f"### 🚇🚌 **{endpoints[0]} → {endpoints[1]}**"
            if _infer_language(user_message, "") == "pt"
            else f"### 🚇🚌 **{endpoints[0]} → {endpoints[1]}**"
        )
        bus_title = "**🚌 Autocarros**" if _infer_language(user_message, "") == "pt" else "**🚌 Buses**"
        metro_title = "**🚇 Metro**"
        timestamp = datetime.now().strftime("%H:%M")
        if fastest_requested:
            first_note = (
                "✅ **Resposta direta:** a melhor alternativa provável é o **Metro**; a Carris fica como opção de superfície."
                if _infer_language(user_message, "") == "pt"
                else "✅ **Direct answer:** the likely best alternative is **Metro**; Carris remains a surface option."
            )
        else:
            first_note = (
                "✅ **Resposta direta:** para não repetir só a Carris, há também uma opção de **Metro de Lisboa** suportada pelos dados."
                if _infer_language(user_message, "") == "pt"
                else "✅ **Direct answer:** instead of only repeating Carris, there is also a **Metro de Lisboa** option supported by the data."
            )
        metro_response = _build_deterministic_metro_route_response(user_message, "") or route_result
        return (
            f"{title}\n\n"
            f"{first_note}\n\n"
            "---\n\n"
            f"{metro_title}\n\n"
            f"{_strip_embedded_transport_route_block(metro_response)}\n\n"
            "---\n\n"
            f"{bus_title}\n\n"
            f"{_strip_embedded_transport_route_block(formatted_carris_response)}\n\n"
            f"📌 **{source_label}:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | [*Carris*](https://www.carris.pt) | **{updated_label}:** {timestamp}"
        )

    if valid_carris_result:
        response = _build_mode_filtered_carris_route_response(
            carris_result,
            user_message,
            endpoints[0],
            endpoints[1],
        )
        return _append_generic_service_area_note(response, raw_destination, area_destination, language)

    return _append_generic_service_area_note(route_result, raw_destination, area_destination, language)


def _route_response_uses_carris_urban(response: str) -> bool:
    """Return whether a route response contains a checked Carris Urban option."""
    if not response:
        return False
    normalized = _normalize_token(response)
    if "carris metropolitana" in normalized:
        return False
    if "carris urban" in normalized or "carris urbana" in normalized:
        return True
    return bool(
        re.search(
            r"(?im)^(?:####?\s*)?(?:🚌\s*)?(?:autocarros|buses|el[eé]tricos|trams)\b|"
            r"\b(?:direct routes found|rotas diretas encontradas)\b|"
            r"\b(?:linha|line)\s+\d{2,3}[A-Z]?\b|"
            r"\*\*\d{2,3}[A-Z]?\*\*",
            response,
        )
    )


def _build_tool_call(name: str, args: dict) -> AIMessage:
    """Creates a deterministic tool call message for the subgraph."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": f"auto_{uuid.uuid4().hex}",
                "type": "tool_call",
            }
        ],
    )


def _build_deterministic_transport_tool_call(user_message: str) -> Optional[AIMessage]:
    """Routes obvious transport coverage prompts to their canonical tool."""
    query = user_message.strip()
    if _query_is_aggregate_transport_status(query):
        language = infer_response_language(user_query=user_message, default="en")
        return _build_tool_call("get_transport_summary", {"language": language})

    if _query_has_route_mode_constraints(query):
        return None
    if _query_has_route_quality_preferences(query) and _extract_route_endpoints(query):
        return None
    if _query_requests_future_cp_schedule(query) or _query_requests_broad_carris_catalog(query):
        return None

    for spec_builder in (
        _build_cp_tool_spec,
        _build_carris_metropolitana_tool_spec,
        _build_metro_tool_spec,
        _build_carris_urban_tool_spec,
    ):
        spec = spec_builder(query)
        if spec:
            if spec.get("name") == "get_carris_metropolitana_alerts":
                spec.setdefault("args", {})["language"] = infer_response_language(
                    user_query=user_message,
                    default="en",
                )
            return _build_tool_call(spec["name"], spec["args"])

    return None


class TransportAgent(BaseAgent):
    """
    Transport specialist agent for Lisbon's public transport.

    Handles:
        - Metro de Lisboa (status, routing, wait times)
        - Carris Urban (city buses and trams: 28E, 15E, 732, etc.)
        - Carris Metropolitana (suburban bus routes, alerts)
        - CP trains (suburban lines: Cascais, Sintra, Azambuja)
        - Multi-modal routing with GPS-based stop finding
    """

    def __init__(self):
        """Initializes the transport agent."""
        super().__init__("transport")
        self.system_prompt = get_transport_prompt(language="en")
        self._system_prompt_dynamic = True
        self._last_transport_context: Optional[dict] = None

    def _get_runtime_system_prompt(self, language: str) -> str:
        """Return the prompt for the requested language while preserving explicit test overrides."""
        if not getattr(self, "_system_prompt_dynamic", False):
            override_prompt = getattr(self, "system_prompt", "")
            if override_prompt:
                return override_prompt

        prompt = get_transport_prompt(language=language)
        self.system_prompt = prompt
        return prompt

    def reset_conversation_context(self) -> None:
        """Clears cached transport follow-up context for the session."""
        self._last_transport_context = None

    def _get_tool_by_name(self, tool_name: str):
        """Returns a loaded tool by name, or None if not found."""
        for tool in self.tools:
            if getattr(tool, "name", "") == tool_name:
                return tool
        return None

    # --- Tool-arg sanitation ----------------------------------------------
    # LLMs sometimes stuff entire conjunctive multi-leg phrases into a single
    # ``origin`` or ``destination`` argument when planning trains/routes, e.g.
    # ``destination="Sete-Rios e como vou depois para Sintra"``. The helpers
    # below strip those trailing clauses before the tool sees them so the
    # underlying API receives a clean station/landmark name.
    _LOCATION_TAIL_SPLITTERS: Tuple[re.Pattern, ...] = (
        re.compile(
            r"\s+e\s+(?:como|que|qual|depois|a\s+seguir|de\s+seguida|a\s+partir)\b.*$",
            re.IGNORECASE,
        ),
        re.compile(r"\s+e\s+(?:depois|a\s+seguir|de\s+seguida)\s+.*$", re.IGNORECASE),
        re.compile(r"\s+and\s+(?:then|how|what|after|next)\b.*$", re.IGNORECASE),
        re.compile(r"\s*[,;]\s*(?:e|and)\s+.*$", re.IGNORECASE),
        re.compile(r"\s*\?\s*.*$"),
    )
    _LOCATION_ARG_KEYS: Tuple[str, ...] = (
        "origin",
        "destination",
        "from_location",
        "to_location",
        "from_station",
        "to_station",
        "near_location_name",
        "location",
        "place_name",
        "station_name",
        "stop_name",
        "start",
        "end",
    )

    @classmethod
    def _clean_location_arg(cls, raw_value: Any) -> Any:
        """Strip multi-leg conjunctive tails from a location-like tool argument."""
        if not isinstance(raw_value, str):
            return raw_value
        cleaned = raw_value.strip()
        if not cleaned:
            return raw_value
        for splitter in cls._LOCATION_TAIL_SPLITTERS:
            cleaned = splitter.sub("", cleaned).strip()
        cleaned = _clean_query_fragment(cleaned)
        cleaned = cleaned.strip(" ,;.-")
        return cleaned or raw_value

    def _preprocess_tool_args(self, tool_name: str, tool_args: dict) -> dict:
        """Sanitize LLM-emitted location arguments before invoking the tool."""
        if not isinstance(tool_args, dict) or not tool_args:
            return tool_args
        cleaned = dict(tool_args)
        for key in self._LOCATION_ARG_KEYS:
            if key in cleaned:
                cleaned[key] = self._clean_location_arg(cleaned[key])
        return cleaned

    @staticmethod
    def _prepend_location_ambiguity(
        response: str,
        tool_args: Dict[str, Any],
        language: str,
    ) -> str:
        """Restore bare-location ambiguity notes after formatter compression."""
        origin = str(tool_args.get("origin") or "").strip()
        destination = str(tool_args.get("destination") or "").strip()
        if not origin or not destination:
            return response
        if "Ambiguidade" in response or "Ambiguity" in response:
            return response

        try:
            from tools.location_resolver import build_location_ambiguity_preamble

            ambiguity_note = build_location_ambiguity_preamble(
                origin,
                destination,
                language=language,
            )
        except Exception:
            ambiguity_note = ""

        if not ambiguity_note:
            return response
        return f"{ambiguity_note}\n\n{response}".strip()

    @staticmethod
    def _extract_follow_up_mode(user_message: str) -> Optional[str]:
        """Extracts a requested transport mode from a short follow-up."""
        query = (user_message or "").lower()
        if any(term in query for term in ["metro"]):
            return "metro"
        if any(term in query for term in ["bus", "autocarro", "autocarros", "carris"]):
            return "bus"
        if any(term in query for term in ["train", "comboio", "comboios", "cp"]):
            return "train"
        if any(term in query for term in ["tram", "trams", "elétrico", "eletrico", "elétricos", "eletricos"]):
            return "tram"
        return None

    @staticmethod
    def _is_referential_mode_follow_up(user_message: str) -> bool:
        """Returns whether a query is a true mode-only follow-up like 'And by metro?'."""
        normalized = re.sub(r"[!?.,;:]+", "", (user_message or "").strip().lower())
        if not normalized:
            return False

        mode_pattern = (
            r"(?:metro|bus|autocarro|autocarros|carris|train|comboio|comboios|cp|"
            r"tram|trams|el[eé]trico|el[eé]tricos)"
        )
        return bool(
            re.fullmatch(
                rf"(?:(?:mas\s+e|e|and|what about|how about|same|also|agora|now)\s+)?"
                rf"(?:(?:de|by)\s+)?{mode_pattern}"
                rf"(?:\s+(?:ou|or|e|and)\s+(?:(?:de|by)\s+)?{mode_pattern})*"
                r"(?:\s+only)?",
                normalized,
            )
        )

    def _rewrite_metro_line_wait_follow_up(self, user_message: str, language: str) -> str:
        """Rewrites short Metro line wait follow-ups using the latest wait-time context."""
        last_context = getattr(self, "_last_transport_context", None) or {}
        if last_context.get("intent") != "metro_line_waits":
            return user_message

        if _extract_route_endpoints(user_message):
            return user_message

        normalized = _normalize_token(user_message)
        explicit_station_list = bool(
            re.search(r"\b(?:quais|listar|lista|list|show|what|which)\b.*\b(?:estacoes|stations)\b", normalized)
        )
        if explicit_station_list and not _query_has_wait_departure_intent(user_message):
            return user_message

        line_id = _extract_metro_line_id(user_message)
        if not line_id and re.search(r"\b(?:mesma|essa|esta|same|that|only|so|apenas)\b", normalized):
            line_id = str(last_context.get("line") or "")

        if not line_id:
            return user_message

        is_short_line_follow_up = bool(line_id and len(normalized.split()) <= 5)
        is_wait_follow_up = is_short_line_follow_up or _query_has_wait_departure_intent(user_message) or bool(
            re.search(r"\b(?:todas?\s+as\s+estacoes|all\s+stations|linha|line|only|so|apenas)\b", normalized)
        )
        if not is_wait_follow_up:
            return user_message

        line_name = _line_display_name(line_id, language)
        if language == "pt":
            return f"Diz-me o tempo de espera de toda a {line_name} do metro em todas as estações."
        return f"Show me the wait times across the whole {line_name} at all stations."

    # Anaphoric destination phrases the user may use to refer to a place
    # already suggested by the assistant in the previous turn(s).
    _ANAPHORIC_DESTINATION_RE = re.compile(
        r"(?:"
        r"o\s+(?:restaurante|almoco|almo[cç]o|jantar|caf[eé]|bar|s[ií]tio|lugar|local|museu|stio|stio)\s+"
        r"(?:que\s+(?:tu\s+)?(?:sugeriste|indicaste|recomendaste|mencionaste|disseste|propuseste|referiste))"
        r"|"
        r"para\s+o\s+(?:restaurante|almoco|almo[cç]o|jantar|caf[eé]|s[ií]tio|lugar|museu)\s+"
        r"(?:que\s+(?:tu\s+)?(?:sugeriste|indicaste|recomendaste|mencionaste|propuseste))"
        r"|"
        r"(?:to|for|at)\s+(?:the|that)\s+(?:restaurant|lunch|dinner|cafe|caf[eé]|bar|place|spot|venue|museum|landmark|attraction)\s+"
        r"(?:you\s+(?:suggested|recommended|mentioned|proposed|said|referred\s+to))"
        r")",
        re.IGNORECASE,
    )

    # Matches the previous assistant context block injected by the graph layer.
    _PREVIOUS_ASSISTANT_CONTEXT_RE = re.compile(
        r"Previous assistant answer[^\n]*:\s*\n(.+?)\Z",
        re.DOTALL,
    )

    # Bold venue name in a list-bullet card, e.g. "- **🍽️ Restaurante Exemplo**".
    # We require a leading list bullet so we do not match section headings such
    # as "**🔵 Locais e atrações**", which would yield bogus destinations.
    _VENUE_CARD_NAME_RE = re.compile(
        r"^\s*[\-*]\s+\*\*[^\w\s]*\s*([A-Z0-9][^\n*]{2,80})\*\*",
        re.MULTILINE,
    )

    def _resolve_anaphoric_destination(
        self, user_message: str, context: str, language: str
    ) -> str:
        """Replace anaphoric destination references with the venue from the previous answer.

        For example:
            "Como vou da origem ate ao restaurante que sugeriste?"
            -> "Como vou da origem ate Restaurante Exemplo?"
        """
        if not context:
            return user_message
        anaphor_match = self._ANAPHORIC_DESTINATION_RE.search(user_message)
        if not anaphor_match:
            return user_message
        ctx_match = self._PREVIOUS_ASSISTANT_CONTEXT_RE.search(context)
        if not ctx_match:
            return user_message
        previous_text = ctx_match.group(1)
        venue_match = self._VENUE_CARD_NAME_RE.search(previous_text)
        if not venue_match:
            return user_message
        venue_name = venue_match.group(1).strip(" .·-")
        if not venue_name or len(venue_name) < 3:
            return user_message
        return (
            user_message[: anaphor_match.start()]
            + venue_name
            + user_message[anaphor_match.end() :]
        )

    def _rewrite_follow_up_transport_query(
        self, user_message: str, language: str, context: str = ""
    ) -> str:
        """Rewrites short transport follow-ups using the last remembered route endpoints."""
        # Resolve anaphoric destinations like "the restaurant you suggested"
        # against the previous assistant answer injected by the graph layer.
        user_message = self._resolve_anaphoric_destination(user_message, context, language)

        if _extract_route_endpoints(user_message):
            return user_message

        metro_wait_follow_up = self._rewrite_metro_line_wait_follow_up(user_message, language)
        if metro_wait_follow_up != user_message:
            return metro_wait_follow_up

        if not self._is_referential_mode_follow_up(user_message):
            return user_message

        last_context = getattr(self, "_last_transport_context", None)
        if not last_context:
            return user_message

        mode = self._extract_follow_up_mode(user_message)
        if not mode:
            return user_message

        origin = str(last_context.get("origin") or "").strip()
        destination = str(last_context.get("destination") or "").strip()
        if not origin or not destination:
            return user_message

        requested_modes = _requested_route_option_modes(user_message)
        if len(requested_modes) > 1:
            mode_order = ["metro", "bus", "train", "tram"]
            ordered_modes = [mode for mode in mode_order if mode in requested_modes]
            if language == "pt":
                labels = {
                    "metro": "metro",
                    "bus": "autocarro",
                    "train": "comboio",
                    "tram": "elétrico",
                }
                mode_labels = [labels[mode] for mode in ordered_modes]
                if len(mode_labels) == 2:
                    mode_phrase = f"de {mode_labels[0]} ou {mode_labels[1]}"
                else:
                    mode_phrase = f"de {', '.join(mode_labels[:-1])} ou {mode_labels[-1]}"
                return f"Como vou de {origin} para {destination} {mode_phrase}?".strip()

            labels = {
                "metro": "metro",
                "bus": "bus",
                "train": "train",
                "tram": "tram",
            }
            mode_labels = [labels[mode] for mode in ordered_modes]
            if len(mode_labels) == 2:
                mode_phrase = f"by {mode_labels[0]} or {mode_labels[1]}"
            else:
                mode_phrase = f"by {', '.join(mode_labels[:-1])} or {mode_labels[-1]}"
            return f"How do I get from {origin} to {destination} {mode_phrase}?".strip()

        if language == "pt":
            mode_phrase = {
                "metro": "de metro",
                "bus": "de autocarro",
                "train": "de comboio",
                "tram": "de elétrico",
            }.get(mode, "")
            return f"Como vou de {origin} para {destination} {mode_phrase}?".strip()

        mode_phrase = {
            "metro": "by metro",
            "bus": "by bus",
            "train": "by train",
            "tram": "by tram",
        }.get(mode, "")
        return f"How do I get from {origin} to {destination} {mode_phrase}?".strip()

    def _remember_transport_context(self, user_message: str) -> None:
        """Caches the latest route endpoints so transport-mode follow-ups can reuse them."""
        line_id = _extract_metro_line_id(user_message)
        if line_id and _query_requests_metro_line_wait_times(user_message):
            self._last_transport_context = {
                "intent": "metro_line_waits",
                "line": line_id,
                "last_user_query": user_message,
                "mode": "metro",
            }
            return

        endpoints = _extract_route_endpoints(user_message)
        if not endpoints:
            return

        self._last_transport_context = {
            "origin": endpoints[0],
            "destination": endpoints[1],
            "last_user_query": user_message,
            "mode": self._extract_follow_up_mode(user_message),
        }

    @staticmethod
    def _is_status_query(user_message: str) -> bool:
        """Detects generic service-status questions that are better answered deterministically."""
        return _query_has_status_intent(user_message)

    def _run_direct_tool_fallback(self, user_message: str) -> Optional[str]:
        """Uses deterministic transport tools for broad status questions."""
        if not self._is_status_query(user_message) or _query_has_wait_departure_intent(user_message):
            return None

        query = user_message.lower()
        language = infer_response_language(user_query=user_message, default="en")
        metro_status_patterns = [
            r"\bis the metro working\b",
            r"\bmetro status\b",
            r"\best[aá] o metro a funcionar\b",
            r"\bestado do metro\b",
            r"\bestado das linhas do metro\b",
            r"\bstatus das linhas do metro\b",
            r"\bperturba(?:cao|ções|coes|ção|caoes)s?\b.*\bmetro\b",
            r"\bperturba(?:cao|ções|coes|ção|caoes)s?\b.*\blinhas\b",
            r"\bdisruptions?\b.*\bmetro\b",
            r"\b(?:delay|delays|problem|problems)\b.*\bmetro\b",
            r"\bmetro\b.*\b(?:delay|delays|problem|problems)\b",
            r"\bmetro de lisboa\b.*\b(?:estado|status|perturba(?:cao|ções|coes|ção|caoes)s?)\b",
            r"\bmetro lines\b",
            r"\bmetro service\b",
        ]

        if any(re.search(pattern, query) for pattern in metro_status_patterns):
            metro_tool = self._get_tool_by_name("get_metro_status")
            if metro_tool:
                raw_status = self._invoke_tool(metro_tool, {}, tool_name="get_metro_status")
                return self._format_deterministic_tool_result(
                    "get_metro_status",
                    {},
                    str(raw_status),
                    language,
                    user_message=user_message,
                )

        summary_tool = self._get_tool_by_name("get_transport_summary")
        if summary_tool:
            return self._invoke_tool(summary_tool, {"language": language}, tool_name="get_transport_summary")

        return None

    def _build_metropolitana_multipart_response(
        self,
        user_message: str,
        language: str,
    ) -> Optional[str]:
        """Answer direct-bus, disruption, and live-location parts in one grounded response."""
        endpoints = _extract_route_endpoints(user_message)
        if not endpoints:
            return None

        normalized = _normalize_token(user_message)
        if not (
            "direct" in normalized
            and any(term in normalized for term in ["disruption", "disruptions", "alert", "alerts"])
            and any(term in normalized for term in ["where", "location", "position", "moment", "now"])
        ):
            return None

        direct_tool = self._get_tool_by_name("find_direct_bus_lines")
        alerts_tool = self._get_tool_by_name("get_carris_metropolitana_alerts")
        live_tool = self._get_tool_by_name("get_bus_realtime_locations")
        if not direct_tool:
            return None

        origin, destination = endpoints
        direct_result = str(
            self._invoke_tool(
                direct_tool,
                {"origin": origin, "destination": destination},
                tool_name="find_direct_bus_lines",
            )
        ).strip()
        has_precise_direct_line = "direct line" in direct_result.lower() and "broad carris metropolitana candidates" not in direct_result.lower()
        line_id = _extract_first_metropolitana_line_id(direct_result) if has_precise_direct_line else None

        alerts_result = ""
        if alerts_tool:
            alerts_result = str(
                self._invoke_tool(
                    alerts_tool,
                    {"area": destination},
                    tool_name="get_carris_metropolitana_alerts",
                )
            ).strip()

        live_result = ""
        if live_tool and line_id:
            live_result = str(
                self._invoke_tool(
                    live_tool,
                    {"line_id": line_id},
                    tool_name="get_bus_realtime_locations",
                )
            ).strip()

        title = f"### 🚌 **Carris Metropolitana: {origin} → {destination}**"
        lines = [title, ""]
        if line_id:
            lines.append(f"- ✅ **Direct bus:** line {line_id} was the first confirmed direct option in the returned data.")
        elif _tool_result_indicates_no_match(direct_result):
            lines.append("- ❌ **Direct bus:** no direct line was confirmed from the available Carris Metropolitana data.")
        else:
            lines.append("- ℹ️ **Direct bus:** directness could not be reduced to one line from the returned data.")

        lines.append(_summarize_relevant_alerts_for_line(alerts_result, line_id))

        if line_id and live_result:
            if "No active buses found" in live_result:
                lines.append(f"- ℹ️ **Live location:** no active live bus is currently reported on line {line_id}.")
            else:
                active_match = re.search(r"found \*\*(?P<count>\d+) active", live_result, flags=re.IGNORECASE)
                count_text = active_match.group("count") if active_match else "live"
                lines.append(f"- 📡 **Live location:** {count_text} active bus{'es' if count_text != '1' else ''} reported on line {line_id}; see the live snapshot below.")
        elif line_id:
            lines.append(f"- ℹ️ **Live location:** no live location feed could be confirmed for line {line_id}.")

        if line_id and live_result and "No active buses found" not in live_result:
            lines.extend(["", "**📡 Live snapshot**"])
            bus_count = 0
            capturing_vehicle = False
            for raw_line in live_result.splitlines():
                stripped = raw_line.strip()
                if stripped.startswith(("- **🚌 Bus", "- 🚌 **Bus")):
                    if bus_count >= 2:
                        break
                    bus_count += 1
                    capturing_vehicle = True
                    lines.append(stripped)
                    continue
                if capturing_vehicle and stripped.startswith("-"):
                    lines.append(f"    {stripped}")

        lines.append("")
        lines.append(self._build_transport_source_line(language, ["[*Carris Metropolitana*](https://www.carrismetropolitana.pt)"]))
        return "\n".join(lines).strip()

    def _build_madeira_nearest_metro_response(
        self,
        user_message: str,
        language: str,
    ) -> Optional[str]:
        """Answer nearest-station queries for the Lisbon Ilha da Madeira address safely."""
        normalized = _normalize_token(user_message)
        if "metro" not in normalized:
            return None
        if not re.search(r"\b(?:nearest|closest|mais proxima|mais proximo|estacao)\b", normalized):
            return None
        if not re.search(r"\b(?:rua|avenida|av)\b.*\bilha da madeira\b|\brua humberto madeira\b", normalized):
            return None

        if language == "pt":
            body = (
                "### 🚇 **Estação de metro mais próxima**\n\n"
                "A estação de referência para a **morada Ilha da Madeira, em Lisboa**, é **Encarnação** (Linha Vermelha).\n\n"
                "- Se te referes à **Ilha da Madeira** enquanto ilha, isso fica fora da rede urbana do Metro de Lisboa.\n"
                "- Para um percurso porta-a-porta, indica também o teu ponto de partida."
            )
        else:
            body = (
                "### 🚇 **Nearest metro station**\n\n"
                "For the **Ilha da Madeira address in Lisbon**, the reference station is **Encarnação** (Red Line).\n\n"
                "- If you mean **Madeira island**, that is outside Lisbon's urban metro network.\n"
                "- For a door-to-door route, also provide your starting point."
            )
        return f"{body}\n\n{self._build_transport_source_line(language, ['[*Metro de Lisboa*](https://www.metrolisboa.pt)'])}"

    def _build_non_station_nearest_metro_response(
        self,
        user_message: str,
        language: str,
    ) -> Optional[str]:
        """Answer POI/neighbourhood nearest-metro questions without dumping the full network."""
        normalized = _normalize_token(user_message)
        if "metro" not in normalized:
            return None
        if not re.search(r"\b(?:nearest|closest|nearby|mais proxima|mais proximo|perto|perto de)\b", normalized):
            return None

        place_match = re.search(
            r"\bis\s+(?P<place>[A-Za-zÀ-ÿ' -]{2,60}?)\s+a\s+metro\s+station\b",
            user_message,
            flags=re.IGNORECASE,
        )
        if not place_match:
            place_match = re.search(
                r"\b(?P<place>[A-Za-zÀ-ÿ' -]{2,60}?)\s+(?:é|e)\s+(?:uma\s+)?esta[cç][aã]o\s+de\s+metro\b",
                user_message,
                flags=re.IGNORECASE,
            )
        if not place_match:
            place_match = re.search(
                r"\b(?:near|nearby|closest to|nearest to|perto de|junto a)\s+(?P<place>[A-Za-zÀ-ÿ' -]{2,60})",
                user_message,
                flags=re.IGNORECASE,
            )
        if not place_match:
            return None

        place_name = place_match.group("place").strip(" .?!,;:")
        if not place_name:
            return None

        try:
            from tools.location_resolver import resolve_location_query

            resolved = resolve_location_query(place_name)
        except Exception:
            return None

        if resolved.get("match_source") == "metro_station":
            return None

        nearest_tool = self._get_tool_by_name("find_nearest_metro")
        if nearest_tool:
            nearest_result = str(
                self._invoke_tool(
                    nearest_tool,
                    {"near_location_name": place_name},
                    tool_name="find_nearest_metro",
                )
            ).strip()
        else:
            from tools.metrolisboa_api import find_nearest_metro

            nearest_result = str(
                find_nearest_metro.invoke({"near_location_name": place_name})
            ).strip()
        nearest_result = re.sub(
            r"(?mis)^###\s+🚇\s+\*\*?Nearest Metro Stations\*\*?\s*\n+",
            "",
            nearest_result,
        ).strip()
        if language != "pt":
            line_name_map = {
                "Amarela": "Yellow",
                "Azul": "Blue",
                "Verde": "Green",
                "Vermelha": "Red",
            }

            def _localize_lines_field(match: re.Match[str]) -> str:
                raw_lines = match.group("value")
                translated = ", ".join(
                    line_name_map.get(part.strip(), part.strip())
                    for part in raw_lines.split(",")
                    if part.strip()
                )
                return f"{match.group('prefix')}{translated}"

            nearest_result = re.sub(
                r"(?m)^(?P<prefix>\s*[-*]\s+🚇\s+\*\*Lines:\*\*\s*)(?P<value>[^\n]+)$",
                _localize_lines_field,
                nearest_result,
            )

        display_name = str(resolved.get("display_name") or place_name).strip()
        if language == "pt":
            intro = (
                f"**{display_name} não é uma estação do Metro de Lisboa.** "
                "Estas são as estações de metro mais próximas que consegui calcular a partir da localização resolvida:"
            )
        else:
            intro = (
                f"**{display_name} is not a Lisbon Metro station.** "
                "These are the nearest metro stations I can calculate from the resolved location:"
            )

        return "\n".join(
            [
                "### 🚇 **Estações de metro mais próximas**" if language == "pt" else "### 🚇 **Nearest Metro Stations**",
                "",
                intro,
                "",
                nearest_result,
                "",
                self._build_transport_source_line(
                    language,
                    ["[*Metro de Lisboa*](https://www.metrolisboa.pt)"],
                ),
            ]
        ).strip()

    @staticmethod
    def _build_transport_source_line(language: str, source_links: List[str]) -> str:
        """Builds a localized transport source line for combined deterministic answers."""
        deduped_sources: List[str] = []
        for source in source_links:
            if source and source not in deduped_sources:
                deduped_sources.append(source)

        timestamp = datetime.now().strftime("%H:%M")
        if language == "pt":
            return f"📌 **Fonte:** {' | '.join(deduped_sources)} | **Atualizado:** {timestamp}"
        return f"📌 **Source:** {' | '.join(deduped_sources)} | **Updated:** {timestamp}"

    @staticmethod
    def _extract_cp_multileg_request(user_message: str) -> Optional[Tuple[str, str, str]]:
        """Extract a two-leg CP request such as Entrecampos → Sete Rios → Sintra."""
        if not re.search(r"\b(cp|comboio|comboios|train|trains)\b", user_message or "", re.IGNORECASE):
            return None
        pattern = re.compile(
            r"\b(?:de|from)\s+(?P<origin>.+?)\s+(?:para|to)\s+(?P<mid>.+?)\s+"
            r"(?:e|and)\s+(?:como\s+(?:vou|ir|chego)\s+)?(?:vou\s+)?(?:depois\s+|a\s+seguir\s+|then\s+|after\s+)?"
            r"(?:para|to)\s+(?P<final>.+?)(?:[\?\!\.,;]|$)",
            re.IGNORECASE,
        )
        match = pattern.search(user_message or "")
        if not match:
            return None
        origin = _resolve_cp_station_name(match.group("origin"))
        intermediate = _resolve_cp_station_name(match.group("mid"))
        final = _resolve_cp_station_name(match.group("final"))
        if not origin or not intermediate or not final:
            return None
        if len({_normalize_token(origin), _normalize_token(intermediate), _normalize_token(final)}) < 3:
            return None
        return origin, intermediate, final

    @staticmethod
    def _strip_nested_cp_trip_block(text: str, *, drop_direct_answer: bool = False) -> str:
        """Remove nested wrapper lines before embedding a CP trip block."""
        kept: List[str] = []
        last_was_rule = False
        for raw_line in (text or "").splitlines():
            stripped = raw_line.strip()
            if not stripped and not kept:
                continue
            if re.match(r"^###\s+🚆", stripped):
                continue
            if re.match(r"^📌\s+\*\*(?:Fonte|Source):\*\*", stripped):
                continue
            if drop_direct_answer and re.match(
                r"^✅\s+\*\*(?:Resposta direta|Direct answer):\*\*",
                stripped,
                flags=re.IGNORECASE,
            ):
                continue
            if stripped == "---":
                if not kept or last_was_rule:
                    continue
                kept.append(raw_line.rstrip())
                last_was_rule = True
                continue
            kept.append(raw_line.rstrip())
            if stripped:
                last_was_rule = False
        while kept and kept[-1].strip() in {"", "---"}:
            kept.pop()
        return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()

    def _build_cp_multileg_response(self, user_message: str, language: str) -> Optional[str]:
        """Build a grounded two-leg CP response instead of passing a compound destination to one tool."""
        legs = self._extract_cp_multileg_request(user_message)
        if not legs:
            return None
        origin, intermediate, final = legs
        train_tool = self._get_tool_by_name("plan_train_trip")
        if not train_tool:
            return None

        try:
            first_raw = str(
                self._invoke_tool(
                    train_tool,
                    {"origin": origin, "destination": intermediate},
                    tool_name="plan_train_trip",
                )
            ).strip()
            second_raw = str(
                self._invoke_tool(
                    train_tool,
                    {"origin": intermediate, "destination": final},
                    tool_name="plan_train_trip",
                )
            ).strip()
        except Exception:
            return None

        first_block = self._strip_nested_cp_trip_block(
            self._format_deterministic_tool_result(
                "plan_train_trip",
                {"origin": origin, "destination": intermediate},
                first_raw,
                language,
                user_message=user_message,
            ),
            drop_direct_answer=True,
        )
        second_block = self._strip_nested_cp_trip_block(
            self._format_deterministic_tool_result(
                "plan_train_trip",
                {"origin": intermediate, "destination": final},
                second_raw,
                language,
                user_message=user_message,
            ),
            drop_direct_answer=True,
        )
        if not first_block and not second_block:
            return None

        if language == "pt":
            title = f"### 🚆 **Comboio CP: {origin} → {intermediate} → {final}**"
            direct = (
                f"✅ **Resposta direta:** primeiro apanha o comboio **{origin} → {intermediate}**; "
                f"depois continua de **{intermediate} → {final}** pela CP suburbana/AML, com as próximas partidas abaixo."
            )
            first_label = f"**1. {origin} → {intermediate}**"
            second_label = f"**2. {intermediate} → {final}**"
        else:
            title = f"### 🚆 **CP train: {origin} → {intermediate} → {final}**"
            direct = (
                f"✅ **Direct answer:** first take **{origin} → {intermediate}**; "
                f"then continue **{intermediate} → {final}** on CP suburban/AML rail, with upcoming departures below."
            )
            first_label = f"**1. {origin} → {intermediate}**"
            second_label = f"**2. {intermediate} → {final}**"

        return "\n".join(
            [
                title,
                "",
                direct,
                "",
                "---",
                "",
                first_label,
                first_block,
                "",
                second_label,
                second_block,
                "",
                self._build_transport_source_line(language, ["[*CP*](https://www.cp.pt)"]),
            ]
        ).strip()

    def _build_train_metro_no_bus_response(
        self,
        user_message: str,
        language: str,
    ) -> Optional[str]:
        """Build a CP + Metro fallback when a direct CP trip is unavailable and buses are excluded."""
        endpoints = _extract_route_endpoints(user_message)
        if not endpoints:
            return None

        normalized = _normalize_token(user_message)
        if not re.search(r"\b(?:cp|comboio|comboios|train|trains)\b", normalized):
            return None
        if not re.search(r"\b(?:sem|evitar|evito|n[aã]o\s+quero|without|avoid|no)\s+(?:autocarro|autocarros|bus|buses)\b", normalized):
            return None
        if _query_strictly_requests_train_only(user_message):
            return _build_mode_unavailable_response(
                mode="comboio" if language == "pt" else "train",
                origin=endpoints[0],
                destination=endpoints[1],
                language=language,
            )

        try:
            from tools.cp_api import get_cp_station_info
            from tools.metrolisboa_api import get_station_lines
        except Exception:
            return None

        origin, destination = endpoints
        origin_cp = get_cp_station_info(origin)
        destination_cp = get_cp_station_info(destination)
        if not origin_cp or not destination_cp:
            return None

        origin_lines = set(origin_cp.get("lines") or [])
        destination_lines = set(destination_cp.get("lines") or [])
        if origin_lines & destination_lines:
            return None

        interchange_candidates = (
            ("Cais do Sodré", "Cais do Sodré", {"cascais"}),
            ("Rossio", "Rossio", {"sintra"}),
            ("Entrecampos", "Entrecampos", {"sintra", "azambuja", "norte"}),
            ("Oriente", "Oriente", {"sintra", "azambuja", "norte"}),
            ("Santa Apolónia", "Santa Apolónia", {"azambuja", "norte"}),
        )

        def metro_station_for_cp_station(cp_info: dict, fallback: str) -> str:
            station_name = str(cp_info.get("name") or fallback).strip()
            aliases = {
                "lisboa - oriente": "Oriente",
                "lisboa oriente": "Oriente",
                "cais do sodre": "Cais do Sodré",
                "santa apolonia": "Santa Apolónia",
                "santa apolónia": "Santa Apolónia",
                "entrecampos": "Entrecampos",
                "rossio": "Rossio",
            }
            return aliases.get(_normalize_token(station_name), station_name)

        destination_metro_station = metro_station_for_cp_station(destination_cp, destination)
        if not get_station_lines(destination_metro_station):
            return None

        selected_candidate: tuple[str, str] | None = None
        for cp_station, metro_station, candidate_lines in interchange_candidates:
            if _normalize_token(cp_station) == _normalize_token(origin):
                continue
            if not (origin_lines & candidate_lines):
                continue
            if not get_station_lines(metro_station):
                continue
            selected_candidate = (cp_station, metro_station)
            break

        if not selected_candidate:
            return None

        transfer_cp_station, transfer_metro_station = selected_candidate
        train_tool = self._get_tool_by_name("plan_train_trip")
        if not train_tool:
            return None

        try:
            raw_train = str(
                self._invoke_tool(
                    train_tool,
                    {"origin": origin, "destination": transfer_cp_station},
                    tool_name="plan_train_trip",
                )
            ).strip()
        except Exception:
            return None
        if not raw_train or _tool_result_indicates_no_match(raw_train):
            return None

        self._record_tool_call(
            "get_route_between_stations",
            {"origin": transfer_metro_station, "destination": destination_metro_station},
        )
        metro_prompt = (
            f"Quero ir de metro entre {transfer_metro_station} e {destination_metro_station}"
            if language == "pt"
            else f"I want to go by metro between {transfer_metro_station} and {destination_metro_station}"
        )
        metro_block = _build_deterministic_metro_route_response(
            user_message=metro_prompt,
            context=f"User language: {language}",
        )
        if not metro_block:
            return None

        train_block = self._strip_nested_cp_trip_block(
            self._format_deterministic_tool_result(
                "plan_train_trip",
                {"origin": origin, "destination": transfer_cp_station},
                raw_train,
                language,
                user_message=user_message,
            ),
            drop_direct_answer=True,
        )
        metro_block = _strip_transport_source_lines(metro_block)
        metro_block = re.sub(
            r"(?m)^###\s+🚇\s+\*\*[^*\n]+\*\*\s*\n+",
            "",
            metro_block,
            count=1,
        ).strip()

        origin_display = _get_transport_display_name(origin)
        destination_display = _get_transport_display_name(destination)
        transfer_display = _get_transport_display_name(transfer_cp_station)
        if language == "pt":
            title = f"### 🚆🚇 **{origin_display} → {destination_display} sem autocarro**"
            direct = (
                f"✅ **Resposta direta:** não há uma ligação CP direta confirmada; "
                f"a melhor opção suportada sem autocarro é **CP até {transfer_display}** "
                f"e depois **Metro até {destination_display}**."
            )
            first_label = f"### 🚆 **CP: {origin_display} → {transfer_display}**"
            second_label = f"### 🚇 **{transfer_metro_station} → {destination_metro_station}**"
        else:
            title = f"### 🚆🚇 **{origin_display} → {destination_display} without bus**"
            direct = (
                f"✅ **Direct answer:** no direct CP link was confirmed; the best supported "
                f"non-bus option is **CP to {transfer_display}** and then **Metro to {destination_display}**."
            )
            first_label = f"### 🚆 **CP: {origin_display} → {transfer_display}**"
            second_label = f"### 🚇 **{transfer_metro_station} → {destination_metro_station}**"

        return "\n".join(
            [
                title,
                "",
                direct,
                "",
                "---",
                "",
                first_label,
                "",
                train_block,
                "",
                "---",
                "",
                second_label,
                "",
                metro_block,
                "",
                self._build_transport_source_line(
                    language,
                    ["[*CP*](https://www.cp.pt)", "[*Metro de Lisboa*](https://www.metrolisboa.pt)"],
                ),
            ]
        ).strip()

    def _build_nearest_cp_destination_response(self, user_message: str, language: str) -> Optional[str]:
        """Build a CP route from a resolved place with a nearby CP station.

        This covers requests such as a university, venue, or address to a CP
        suburban destination. The first leg is stated as an access leg to the
        nearest CP station; the train leg still comes from ``plan_train_trip``.
        """
        endpoints = _extract_route_endpoints(user_message)
        if not endpoints:
            return None

        preferences = _parse_route_mode_preferences(user_message)
        if preferences["metro_only"] or preferences["bus_only"] or preferences["tram_only"]:
            return None
        if re.search(
            r"\b(?:sem|evitar|n[aã]o\s+quero|without|avoid|no)\s+(?:cp|comboio|comboios|train|trains)\b",
            user_message,
            flags=re.IGNORECASE,
        ):
            return None

        origin, destination = endpoints
        try:
            from tools.cp_api import get_cp_station_info
            from tools.metrolisboa_api import get_landmark_info
        except Exception:
            return None

        destination_cp = get_cp_station_info(destination)
        if not destination_cp:
            return None

        origin_landmark = get_landmark_info(origin) or {}
        train_station = str(origin_landmark.get("train_station") or "").strip()
        if not train_station:
            if _query_strictly_requests_train_only(user_message):
                return _build_mode_unavailable_response(
                    mode="comboio" if language == "pt" else "train",
                    origin=origin,
                    destination=destination,
                    language=language,
                )
            try:
                from tools.transport_api import get_route_between_stations

                route_result = str(
                    get_route_between_stations.invoke(
                        {"origin": origin, "destination": destination}
                    )
                ).strip()
            except Exception:
                route_result = ""
            if route_result and _route_result_is_metro_only_partial(route_result):
                bus_requested = bool(_requested_route_option_modes(user_message) & {"bus"})
                if not bus_requested:
                    cp_bridge = _build_cp_bridge_for_partial_metro_route(
                        user_message=user_message,
                        origin=origin,
                        destination=destination,
                        route_result=route_result,
                        requested_bus=False,
                        checked_metropolitana_direct=True,
                    )
                    if cp_bridge:
                        self._record_tool_call(
                            "get_route_between_stations",
                            {"origin": origin, "destination": destination},
                        )
                        return cp_bridge
                metropolitana_bridge = _build_metropolitana_bridge_for_partial_metro_route(
                    user_message=user_message,
                    origin=origin,
                    destination=destination,
                    route_result=route_result,
                )
                if metropolitana_bridge:
                    self._record_tool_call(
                        "get_route_between_stations",
                        {"origin": origin, "destination": destination},
                    )
                    return metropolitana_bridge
                cp_bridge = _build_cp_bridge_for_partial_metro_route(
                    user_message=user_message,
                    origin=origin,
                    destination=destination,
                    route_result=route_result,
                    requested_bus=bus_requested,
                    checked_metropolitana_direct=True,
                )
                if cp_bridge:
                    self._record_tool_call(
                        "get_route_between_stations",
                        {"origin": origin, "destination": destination},
                    )
                    return cp_bridge
            return None
        if _normalize_token(train_station) == _normalize_token(origin):
            return None

        train_tool = self._get_tool_by_name("plan_train_trip")
        if not train_tool:
            return None

        destination_station = str(destination_cp.get("name") or destination).strip()
        try:
            raw_trip = str(
                self._invoke_tool(
                    train_tool,
                    {"origin": train_station, "destination": destination_station},
                    tool_name="plan_train_trip",
                )
            ).strip()
        except Exception:
            return None
        if not raw_trip or _tool_result_indicates_no_match(raw_trip):
            return None

        trip_block = self._strip_nested_cp_trip_block(
            self._format_deterministic_tool_result(
                "plan_train_trip",
                {"origin": train_station, "destination": destination_station},
                raw_trip,
                language,
                user_message=user_message,
            ),
            drop_direct_answer=True,
        )
        if not trip_block:
            return None

        origin_display = _get_transport_display_name(origin)
        destination_display = _get_transport_display_name(destination)
        walk_minutes = origin_landmark.get("train_walk_minutes")

        if language == "pt":
            access_line = (
                f"- 🚶 **{origin_display} → {train_station}:** cerca de {walk_minutes} min a pé até à estação CP."
                if walk_minutes
                else f"- 🚶 **{origin_display} → {train_station}:** começa por chegar à estação CP mais próxima."
            )
            direct = (
                f"✅ **Resposta direta:** vai primeiro até **{train_station}** e depois apanha a **CP suburbana/AML** "
                f"para **{destination_display}**."
            )
            title = f"### 🚆 **{origin_display} → {destination_display}**"
            flow_title = "🧭 **Sequência recomendada**"
        else:
            access_line = (
                f"- 🚶 **{origin_display} → {train_station}:** about {walk_minutes} min on foot to the CP station."
                if walk_minutes
                else f"- 🚶 **{origin_display} → {train_station}:** first reach the nearest CP station."
            )
            direct = (
                f"✅ **Direct answer:** first reach **{train_station}**, then take **CP suburban/AML rail** "
                f"to **{destination_display}**."
            )
            title = f"### 🚆 **{origin_display} → {destination_display}**"
            flow_title = "🧭 **Recommended sequence**"

        return "\n".join(
            [
                title,
                "",
                direct,
                "",
                "---",
                "",
                flow_title,
                access_line,
                f"- 🚆 **{train_station} → {destination_station}:** CP suburbana/AML.",
                "",
                "---",
                "",
                trip_block,
                "",
                self._build_transport_source_line(language, ["[*CP*](https://www.cp.pt)"]),
            ]
        ).strip()

    @staticmethod
    def _extract_train_lines_summary(summary_text: str) -> Optional[str]:
        """Extracts the CP line summary from a train-tool response."""
        if not summary_text:
            return None

        match = re.search(r"(?:Linhas?|Line(?:s)?):\s*\**([^*\n]+)\**", summary_text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _is_mode_comparison_query(user_message: str) -> bool:
        """Detect explicit transport-mode comparison requests such as metro vs train."""
        normalized_query = unicodedata.normalize("NFKD", user_message or "")
        normalized_query = normalized_query.encode("ascii", "ignore").decode("ascii").lower()
        compares_metro_train = (
            "metro" in normalized_query
            and any(term in normalized_query for term in ["comboio", "comboios", "train", "trains"])
        )
        compares_bus_train = (
            any(term in normalized_query for term in ["autocarro", "autocarros", "bus", "buses"])
            and any(term in normalized_query for term in ["comboio", "comboios", "train", "trains"])
        )
        asks_comparison = bool(
            re.search(r"\b(mais rapid[oa]|mais barat[oa]|faster|fastest|cheaper|cheapest|compare|comparar)\b", normalized_query)
        )
        return (compares_metro_train or compares_bus_train) and asks_comparison

    @staticmethod
    def _extract_duration_minutes(summary_text: str) -> Optional[int]:
        """Extract a best-effort trip duration in minutes from a transport summary."""
        if not summary_text:
            return None

        normalized_text = unicodedata.normalize("NFKD", summary_text)
        normalized_text = normalized_text.encode("ascii", "ignore").decode("ascii")
        patterns = [
            r"(?:Tempo total estimado|Estimated total time|Duracao|Duration):\s*\**~?\s*(\d+)",
            r"~\s*(\d+)\s*min",
            r"\b(\d+)\s*min(?:utos?)?\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized_text, flags=re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _extract_departure_times(summary_text: str, limit: int = 3) -> List[str]:
        """Extract the first distinct departure times listed in a summary."""
        if not summary_text:
            return []

        matches = re.findall(r"(?:\*\*)?(\d{1,2}:\d{2})(?:\*\*)?\s*(?:→|>)", summary_text)
        deduped: List[str] = []
        for item in matches:
            if item not in deduped:
                deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    @staticmethod
    def _extract_metro_route_bullets(metro_response: str) -> List[str]:
        """Extract the concrete Metro route bullets from a deterministic route response."""
        if not metro_response:
            return []

        bullets: List[str] = []
        in_route_section = False
        for raw_line in metro_response.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                if in_route_section and bullets:
                    break
                continue
            if "O seu Trajeto de Metro" in stripped or "Your Metro Route" in stripped:
                in_route_section = True
                continue
            if not in_route_section:
                continue
            if stripped.startswith(("🗓️", "💡", "📌", "⚠️")):
                break
            if stripped.startswith("-"):
                bullets.append(stripped)

        return bullets

    @staticmethod
    def _clean_nested_mode_option(response: str) -> str:
        """Remove nested title/footer noise from an embedded mode-option answer."""
        cleaned = _strip_transport_source_lines(str(response or "")).strip()
        cleaned = re.sub(r"(?m)^###\s+[^\n]+\n*", "", cleaned).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    def _build_bus_train_comparison_response(
        self,
        *,
        user_message: str,
        context: str,
        origin: str,
        destination: str,
        language: str,
    ) -> Optional[str]:
        """Compare bus and train options without collapsing the request to CP only."""
        bus_query = (
            f"Quero ir de autocarro de {origin} para {destination}."
            if language == "pt"
            else f"I want to go by bus from {origin} to {destination}."
        )
        bus_response = self._build_mode_constrained_route_response(
            user_message=bus_query,
            context=context,
            language=language,
        ) or ""

        train_query = (
            f"Quero ir de {origin} para {destination}."
            if language == "pt"
            else f"I want to go from {origin} to {destination}."
        )
        train_response = self._build_nearest_cp_destination_response(train_query, language) or ""
        if not train_response:
            train_tool = self._get_tool_by_name("plan_train_trip")
            if train_tool:
                raw_train = str(
                    self._invoke_tool(
                        train_tool,
                        {"origin": origin, "destination": destination},
                        tool_name="plan_train_trip",
                    )
                ).strip()
                if raw_train:
                    train_response = self._format_deterministic_tool_result(
                        tool_name="plan_train_trip",
                        tool_args={"origin": origin, "destination": destination},
                        result=raw_train,
                        language=language,
                        user_message=user_message,
                    )

        if not bus_response and not train_response:
            return None

        cleaned_bus = self._clean_nested_mode_option(bus_response)
        cleaned_train = self._clean_nested_mode_option(train_response)
        cleaned_train = re.sub(
            r"(?mis)\n+---\n+\s*(?:###\s+)?🚌\s+\*\*(?:Autocarro|Bus)\*\*[\s\S]*$",
            "",
            cleaned_train,
        ).strip()
        cleaned_train = re.split(
            r"(?mis)\n+\s*(?:---\s*\n+)?(?:#{1,6}\s+)?(?:\*\*)?🚌\s+(?:Autocarro|Bus)(?:\*\*)?.*$",
            cleaned_train,
            maxsplit=1,
        )[0].strip()
        cleaned_train = re.sub(
            r"(?m)^###\s+🚆\s+\*\*(?:Comboio / CP|Train / CP)\*\*$",
            "#### 🚆 **Horários CP**" if language == "pt" else "#### 🚆 **CP Timetable**",
            cleaned_train,
        )
        cleaned_train = re.sub(
            r"(?m)^\*\*🚆\s+(?:Comboio / CP|Train / CP)\*\*$",
            "#### 🚆 **Horários CP**" if language == "pt" else "#### 🚆 **CP Timetable**",
            cleaned_train,
        )
        normalized_bus = _normalize_token(cleaned_bus)
        normalized_train = _normalize_token(cleaned_train)
        bus_confirmed = bool(cleaned_bus and not re.search(r"\b(?:nao consegui|couldn't confirm|could not confirm)\b", normalized_bus))
        train_confirmed = bool(cleaned_train and not re.search(r"\b(?:nao consegui|couldn't confirm|could not confirm)\b", normalized_train))

        origin_display = _get_transport_display_name(origin)
        destination_display = _get_transport_display_name(destination)
        arrow = "\u2192"
        if language == "pt":
            title = f"### 🚌🚆 **{origin_display} {arrow} {destination_display}**"
            direct = (
                "✅ **Resposta direta:** comparei autocarro e comboio com os dados disponíveis."
            )
            bus_heading = "### 🚌 **Autocarro**"
            train_heading = "### 🚆 **Comboio / CP**"
            if train_confirmed and not bus_confirmed:
                recommendation = "- **Melhor opção suportada:** comboio/CP, com o acesso inicial indicado na opção ferroviária."
            elif bus_confirmed and not train_confirmed:
                recommendation = "- **Melhor opção suportada:** autocarro, porque não consegui confirmar uma ligação ferroviária aplicável."
            elif bus_confirmed and train_confirmed:
                recommendation = "- **Melhor opção:** compara a próxima partida real; ambas têm dados suficientes para serem consideradas."
            else:
                recommendation = "- **Melhor opção:** não consigo escolher com segurança porque nenhuma alternativa ficou totalmente confirmada."
            unavailable_bus = f"- ⚠️ Não consegui confirmar uma opção de autocarro entre **{origin_display}** e **{destination_display}**."
            unavailable_train = f"- ⚠️ Não consegui confirmar uma opção ferroviária entre **{origin_display}** e **{destination_display}**."
            source_line = self._build_transport_source_line(
                language,
                [
                    "[*Carris*](https://www.carris.pt)",
                    "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
                    "[*CP*](https://www.cp.pt)",
                ],
            )
        else:
            title = f"### 🚌🚆 **{origin_display} {arrow} {destination_display}**"
            direct = "✅ **Direct answer:** I compared bus and train using the available data."
            bus_heading = "### 🚌 **Bus**"
            train_heading = "### 🚆 **Train / CP**"
            if train_confirmed and not bus_confirmed:
                recommendation = "- **Best supported option:** train/CP, with the initial access leg shown in the rail option."
            elif bus_confirmed and not train_confirmed:
                recommendation = "- **Best supported option:** bus, because I could not confirm an applicable rail connection."
            elif bus_confirmed and train_confirmed:
                recommendation = "- **Best option:** compare the next real departure; both options have enough data to consider."
            else:
                recommendation = "- **Best option:** I cannot choose safely because neither alternative was fully confirmed."
            unavailable_bus = f"- ⚠️ I could not confirm a bus option between **{origin_display}** and **{destination_display}**."
            unavailable_train = f"- ⚠️ I could not confirm a rail option between **{origin_display}** and **{destination_display}**."
            source_line = self._build_transport_source_line(
                language,
                [
                    "[*Carris*](https://www.carris.pt)",
                    "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
                    "[*CP*](https://www.cp.pt)",
                ],
            )

        return "\n".join(
            [
                title,
                "",
                direct,
                "",
                "---",
                "",
                bus_heading,
                "",
                cleaned_bus or unavailable_bus,
                "",
                "---",
                "",
                train_heading,
                "",
                cleaned_train or unavailable_train,
                "",
                "---",
                "",
                "### ✅ **Recomendação**" if language == "pt" else "### ✅ **Recommendation**",
                "",
                recommendation,
                "",
                source_line,
            ]
        ).strip()

    def _build_mode_comparison_response(
        self,
        user_message: str,
        context: str = "",
        language: str = "en",
    ) -> Optional[str]:
        """Build a deterministic metro-vs-train comparison answer when the user asks for one."""
        if not self._is_mode_comparison_query(user_message):
            return None

        endpoints = _extract_route_endpoints(user_message)
        if not endpoints:
            return None

        origin, destination = endpoints
        requested_modes = _requested_route_option_modes(user_message)
        if {"bus", "train"}.issubset(requested_modes) and "metro" not in requested_modes:
            return self._build_bus_train_comparison_response(
                user_message=user_message,
                context=context,
                origin=origin,
                destination=destination,
                language=language,
            )

        metro_response = _build_deterministic_metro_route_response(user_message, context) or ""
        if metro_response:
            self._record_tool_call(
                "get_route_between_stations",
                {"origin": origin, "destination": destination},
            )

        train_tool = self._get_tool_by_name("plan_train_trip")
        train_response = ""
        if train_tool:
            train_response = str(
                self._invoke_tool(
                    train_tool,
                    {"origin": origin, "destination": destination},
                    tool_name="plan_train_trip",
                )
            ).strip()

        if not metro_response and not train_response:
            return None

        metro_minutes = self._extract_duration_minutes(metro_response)
        metro_route_bullets = self._extract_metro_route_bullets(metro_response)
        train_minutes = self._extract_duration_minutes(train_response)
        train_lines = self._extract_train_lines_summary(train_response)
        train_departures = self._extract_departure_times(train_response, limit=3)
        normalized_query = unicodedata.normalize("NFKD", user_message or "")
        normalized_query = normalized_query.encode("ascii", "ignore").decode("ascii").lower()
        asks_cheapest = bool(re.search(r"\b(mais barat[oa]|cheaper|cheapest|preco|price)\b", normalized_query))

        arrow = "\u2192"
        if language == "pt":
            lines = [
                f"**Compara\u00e7\u00e3o:** {origin} {arrow} {destination}",
                "",
                "**\U0001F687 Metro de Lisboa**",
                f"\u23F1\uFE0F **Tempo estimado:** {metro_minutes} min"
                if metro_minutes is not None
                else "\u23F1\uFE0F **Tempo estimado:** n\u00e3o foi poss\u00edvel confirmar com os dados dispon\u00edveis",
            ]
            if "circulação normal" in metro_response.lower():
                lines.append("✅ **Estado:** linhas de Metro usadas sem perturbação reportada no momento da consulta")
            if metro_route_bullets:
                lines.extend(["🧭 **Trajeto Metro:**", *metro_route_bullets])
                if "sete rios" in _normalize_token(destination) and any(
                    "Jardim Zoológico" in bullet or "Jardim Zoologico" in bullet
                    for bullet in metro_route_bullets
                ):
                    lines.extend(["", "ℹ️ **Sete Rios no Metro:** a estação que serve Sete Rios chama-se **Jardim Zoológico**."])
            else:
                lines.append("⚠️ **Trajeto Metro:** não foi possível confirmar as linhas e saídas com os dados disponíveis")
            lines.extend([
                "",
                "---",
                "",
                "**\U0001F686 Comboio**",
                f"\u23F1\uFE0F **Tempo estimado:** {train_minutes} min"
                if train_minutes is not None
                else "\u23F1\uFE0F **Tempo estimado:** n\u00e3o foi poss\u00edvel confirmar com os dados dispon\u00edveis",
                f"📍 **Percurso:** embarca em **{origin}** e sai em **{destination}**",
                "🚆 **Ligação:** direta nas partidas mostradas",
            ])
            if "sem dados em tempo real" in train_response.lower():
                lines.append("📡 **Tempo real CP:** sem dados em tempo real no feed usado")
            if train_lines:
                lines.append(f"\U0001F686 **Linhas:** {train_lines}")
            if train_departures:
                lines.append(f"\U0001F550 **Pr\u00f3ximas sa\u00eddas mostradas:** {', '.join(train_departures)}")

            lines.extend(["", "---", "", "**\u2705 Conclus\u00e3o**"])
            if metro_minutes is not None and train_minutes is not None:
                faster_label = "Comboio" if train_minutes < metro_minutes else "Metro de Lisboa"
                lines.append(f"- **Mais r\u00e1pido:** {faster_label}")
            else:
                lines.append("- **Mais r\u00e1pido:** n\u00e3o foi poss\u00edvel comparar com seguran\u00e7a porque falta pelo menos uma dura\u00e7\u00e3o oficial")
            if asks_cheapest:
                lines.append("- **Mais barato:** não foi possível confirmar com dados oficiais de tarifa nas fontes disponíveis")
        else:
            lines = [
                f"**Comparison:** {origin} {arrow} {destination}",
                "",
                "**\U0001F687 Lisbon Metro**",
                f"\u23F1\uFE0F **Estimated time:** {metro_minutes} min"
                if metro_minutes is not None
                else "\u23F1\uFE0F **Estimated time:** could not be confirmed from the available data",
            ]
            if "normal service" in metro_response.lower() or "circulação normal" in metro_response.lower():
                lines.append("✅ **Status:** Metro lines used have no reported disruption at query time")
            if metro_route_bullets:
                lines.extend(["🧭 **Metro route:**", *metro_route_bullets])
                if "sete rios" in _normalize_token(destination) and any(
                    "Jardim Zoológico" in bullet or "Jardim Zoologico" in bullet
                    for bullet in metro_route_bullets
                ):
                    lines.extend(["", "ℹ️ **Sete Rios by Metro:** the Metro station serving Sete Rios is **Jardim Zoológico**."])
            else:
                lines.append("⚠️ **Metro route:** lines and exits could not be confirmed from the available data")
            lines.extend([
                "",
                "---",
                "",
                "**\U0001F686 Train**",
                f"\u23F1\uFE0F **Estimated time:** {train_minutes} min"
                if train_minutes is not None
                else "\u23F1\uFE0F **Estimated time:** could not be confirmed from the available data",
                f"📍 **Route:** board at **{origin}** and exit at **{destination}**",
                "🚆 **Connection:** direct on the listed departures",
            ])
            if "sem dados em tempo real" in train_response.lower() or "no real-time" in train_response.lower():
                lines.append("📡 **CP real time:** no real-time data in the feed used")
            if train_lines:
                lines.append(f"\U0001F686 **Lines:** {train_lines}")
            if train_departures:
                lines.append(f"\U0001F550 **Next departures shown:** {', '.join(train_departures)}")

            lines.extend(["", "---", "", "**\u2705 Verdict**"])
            if metro_minutes is not None and train_minutes is not None:
                faster_label = "Train" if train_minutes < metro_minutes else "Lisbon Metro"
                lines.append(f"- **Faster:** {faster_label}")
            else:
                lines.append("- **Faster:** I could not compare confidently because at least one official duration is missing")
            if asks_cheapest:
                lines.append("- **Cheaper:** official fare data could not be confirmed with the currently available sources")

        lines.extend([
            "",
            self._build_transport_source_line(
                language,
                [
                    "[*Metro de Lisboa*](https://www.metrolisboa.pt)",
                    "[*CP*](https://www.cp.pt)",
                ],
            ),
        ])
        return "\n".join(lines).strip()

    def _format_deterministic_tool_result(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: str,
        language: str,
        user_message: str = "",
    ) -> str:
        """Post-process deterministic single-tool outputs for cleaner user rendering."""
        cleaned_result = str(result or "").strip()
        if not cleaned_result:
            return cleaned_result

        if tool_name == "get_route_between_stations":
            origin = str(tool_args.get("origin") or "").strip()
            destination = str(tool_args.get("destination") or "").strip()
            if origin and destination and _route_result_is_metro_only_partial(cleaned_result):
                broad_destination = _bare_aml_municipality_label(destination)
                bus_requested = bool(_requested_route_option_modes(user_message) & {"bus"})
                if broad_destination and not bus_requested:
                    cp_bridge = _build_cp_bridge_for_partial_metro_route(
                        user_message=user_message,
                        origin=origin,
                        destination=destination,
                        route_result=cleaned_result,
                        checked_metropolitana_direct=True,
                    )
                    if cp_bridge:
                        return cp_bridge
                metropolitana_bridge = _build_metropolitana_bridge_for_partial_metro_route(
                    user_message=user_message,
                    origin=origin,
                    destination=destination,
                    route_result=cleaned_result,
                )
                if metropolitana_bridge:
                    return metropolitana_bridge
                if not (broad_destination and not bus_requested):
                    cp_bridge = _build_cp_bridge_for_partial_metro_route(
                        user_message=user_message,
                        origin=origin,
                        destination=destination,
                        route_result=cleaned_result,
                        checked_metropolitana_direct=True,
                    )
                    if cp_bridge:
                        return cp_bridge

        if tool_name == "get_metro_line_wait_times":
            raw_line = str(tool_args.get("line") or user_message or "").strip()
            line_id = _extract_metro_line_id(raw_line) or _extract_metro_line_id(user_message)
            if line_id:
                return _format_metro_line_wait_snapshot(
                    line_id=line_id,
                    wait_result=cleaned_result,
                    language=language,
                    user_message=user_message,
                )

        if tool_name == "get_all_metro_stations":
            return _format_all_metro_stations(language=language, user_message=user_message)

        if tool_name == "plan_train_trip":
            origin = str(tool_args.get("origin") or "Origem" if language == "pt" else tool_args.get("origin") or "Origin").strip()
            destination = str(
                tool_args.get("destination") or "Destino" if language == "pt" else tool_args.get("destination") or "Destination"
            ).strip()
            title_match = re.search(
                r"^###\s*🚆\s*\*\*Comboio:\s*(?P<origin>.+?)\s*→\s*(?P<destination>.+?)\*\*",
                cleaned_result,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if title_match:
                origin = title_match.group("origin").strip()
                destination = title_match.group("destination").strip()

            if _tool_result_indicates_no_match(cleaned_result):
                if language == "pt":
                    return "\n".join(
                        [
                            f"### 🚆 **{origin} → {destination}**",
                            "",
                            "✅ **Resposta direta:** não consegui confirmar uma ligação direta de **CP suburbano/AML** para este par de estações nos dados disponíveis.",
                            "",
                            "---",
                            "",
                            "💡 **O que isto significa:**",
                            "- 🚆 A cobertura confirmada do LISBOA para CP centra-se em comboios suburbanos/AML.",
                            "- 🧭 Serviços de longo curso ou destinos fora da AML ficam fora do âmbito confirmado.",
                        ]
                    ).strip()
                return "\n".join(
                    [
                        f"### 🚆 **{origin} → {destination}**",
                        "",
                        "✅ **Direct answer:** I could not confirm a direct **CP suburban/AML** train connection for this station pair with the available data.",
                        "",
                        "---",
                        "",
                        "💡 **What this means:**",
                        "- 🚆 LISBOA's confirmed CP coverage is focused on suburban/AML rail data.",
                        "- 🧭 Long-distance services or destinations outside the AML are outside the confirmed scope.",
                    ]
                ).strip()

            no_more_match = re.search(
                r"No more trains today from \*\*(?P<origin>.+?)\*\* to \*\*(?P<destination>.+?)\*\*"
                r".*?There are (?P<count>\d+) trips on other days",
                cleaned_result,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if no_more_match:
                origin = no_more_match.group("origin").strip() or origin
                destination = no_more_match.group("destination").strip() or destination
                other_days_count = no_more_match.group("count").strip()
                line_value = _infer_cp_line_label_for_pair(origin, destination, language)
                if language == "pt":
                    lines = [
                        f"### 🚆 **{origin} → {destination}**",
                        "",
                        "✅ **Resposta direta:** não há mais comboios confirmados hoje para esta ligação CP suburbana/AML.",
                        "",
                        "---",
                        "",
                        "### 📊 **Resumo da viagem**",
                        "- 🚆 **Operador:** CP suburbano/AML",
                    ]
                    if line_value:
                        lines.append(f"- 🛤️ **Linha:** {line_value}")
                    lines.extend(
                        [
                            "- ⏰ **Próximo comboio hoje:** sem mais partidas confirmadas",
                            f"- 📊 **Serviço noutros dias:** {other_days_count} viagens nos dados de horário disponíveis",
                            "",
                            "💡 **Antes de sair:** confirma no site/app da CP a primeira partida disponível para a data em que vais viajar.",
                        ]
                    )
                    return "\n".join(lines).strip()

                lines = [
                    f"### 🚆 **{origin} → {destination}**",
                    "",
                    "✅ **Direct answer:** there are no more confirmed trains today for this CP suburban/AML connection.",
                    "",
                    "---",
                    "",
                    "### 📊 **Trip summary**",
                    "- 🚆 **Operator:** CP suburban/AML",
                ]
                if line_value:
                    lines.append(f"- 🛤️ **Line:** {line_value}")
                lines.extend(
                    [
                        "- ⏰ **Next train today:** no more confirmed departures",
                        f"- 📊 **Service on other days:** {other_days_count} trips in the available timetable data",
                        "",
                        "💡 **Before leaving:** check the CP website/app for the first available departure on your travel date.",
                    ]
                )
                return "\n".join(lines).strip()

            field_values: dict[str, str] = {}
            for raw_line in cleaned_result.splitlines():
                stripped = raw_line.strip()
                field_match = re.match(
                    r"^-\s*(?:[^\w*]+\s*)?\*\*(?P<label>[^:*]+):\*\*\s*(?P<value>.+)$",
                    stripped,
                )
                if field_match:
                    label = _normalize_token(field_match.group("label"))
                    field_values[label] = field_match.group("value").strip().rstrip(".")

            departures: list[tuple[str, str, str, str]] = []
            more_departures = ""
            for raw_line in cleaned_result.splitlines():
                stripped = raw_line.strip()
                departure_match = re.search(
                    r"\*\*(?P<departure>\d{2}:\d{2})\*\*\s*→\s*(?P<arrival>\d{2}:\d{2})\s*\((?P<duration>[^)]+)\)(?:\s*—\s*(?P<route>.+))?",
                    stripped,
                )
                if departure_match:
                    departures.append(
                        (
                            departure_match.group("departure"),
                            departure_match.group("arrival"),
                            departure_match.group("duration").strip(),
                            (departure_match.group("route") or "").strip(),
                        )
                    )
                    continue
                if stripped.startswith("- ..."):
                    more_departures = stripped.lstrip("- ").strip()

            line_value = field_values.get("linha") or field_values.get("linhas")
            duration_value = field_values.get("duracao") or field_values.get("duração")
            status_value = field_values.get("estado")
            remaining_value = field_values.get("partidas restantes hoje")
            route_noun = "Linhas" if field_values.get("linhas") else "Linha"

            next_departure = departures[0][0] if departures else ""
            if language == "pt":
                if line_value and next_departure:
                    direct_answer = (
                        f"segue de **CP suburbano/AML** pela **{line_value}**; "
                        f"a próxima partida mostrada sai às **{next_departure}**."
                    )
                elif line_value:
                    direct_answer = f"segue de **CP suburbano/AML** pela **{line_value}**."
                else:
                    direct_answer = "segue pela ligação CP suburbana/AML confirmada nos dados disponíveis."

                lines = [
                    f"### 🚆 **{origin} → {destination}**",
                    "",
                    f"✅ **Resposta direta:** {direct_answer}",
                    "",
                    "---",
                    "",
                    "📊 **Resumo da viagem**",
                    "- 🚆 **Operador:** CP suburbano/AML",
                ]
                if line_value:
                    lines.append(f"- 🛤️ **{route_noun}:** {line_value}")
                if duration_value:
                    lines.append(f"- ⏱️ **Duração:** {duration_value}")
                if status_value:
                    status_emoji = "🚦" if "atras" in _normalize_token(status_value) else "✅" if "horas" in _normalize_token(status_value) else "ℹ️"
                    lines.append(f"- {status_emoji} **Estado:** {status_value}")
                if remaining_value:
                    lines.append(f"- 📊 **Partidas restantes hoje:** {remaining_value}")
                if departures:
                    lines.extend(["", "---", "", "🕐 **Próximas partidas**"])
                    for departure, arrival, duration, route in departures[:3]:
                        suffix = f" — {route}" if route else ""
                        lines.append(f"- **{departure}** → {arrival} ({duration}){suffix}")
                    if more_departures:
                        lines.append(f"- {more_departures}")
                lines.extend(
                    [
                        "",
                        "💡 **Antes de sair:** confirma a plataforma e a partida no momento da viagem.",
                    ]
                )
                return "\n".join(lines).strip()

            if line_value and next_departure:
                direct_answer = (
                    f"take a **CP suburban/AML** train on **{line_value}**; "
                    f"the next listed departure leaves at **{next_departure}**."
                )
            elif line_value:
                direct_answer = f"take a **CP suburban/AML** train on **{line_value}**."
            else:
                direct_answer = "use the CP suburban/AML connection confirmed in the available data."

            lines = [
                f"### 🚆 **{origin} → {destination}**",
                "",
                f"✅ **Direct answer:** {direct_answer}",
                "",
                "---",
                "",
                "📊 **Trip summary**",
                "- 🚆 **Operator:** CP suburban/AML",
            ]
            if line_value:
                english_route_noun = "Lines" if route_noun == "Linhas" else "Line"
                lines.append(f"- 🛤️ **{english_route_noun}:** {line_value}")
            if duration_value:
                lines.append(f"- ⏱️ **Duration:** {duration_value}")
            if status_value:
                status_emoji = "🚦" if "delay" in _normalize_token(status_value) or "atras" in _normalize_token(status_value) else "✅" if "on time" in status_value.lower() else "ℹ️"
                lines.append(f"- {status_emoji} **Status:** {status_value}")
            if remaining_value:
                lines.append(f"- 📊 **Departures left today:** {remaining_value}")
            if departures:
                lines.extend(["", "---", "", "🕐 **Next departures**"])
                for departure, arrival, duration, route in departures[:3]:
                    suffix = f" — {route}" if route else ""
                    lines.append(f"- **{departure}** → {arrival} ({duration}){suffix}")
                if more_departures:
                    lines.append(f"- {more_departures}")
            lines.extend(["", "💡 **Before leaving:** confirm the platform and departure at travel time."])
            return "\n".join(lines).strip()

        if tool_name == "get_metro_status":
            line_name_map = {
                "yellow line": ("Linha Amarela", "Yellow Line", "🟡"),
                "blue line": ("Linha Azul", "Blue Line", "🔵"),
                "green line": ("Linha Verde", "Green Line", "🟢"),
                "red line": ("Linha Vermelha", "Red Line", "🔴"),
                "linha amarela": ("Linha Amarela", "Yellow Line", "🟡"),
                "linha azul": ("Linha Azul", "Blue Line", "🔵"),
                "linha verde": ("Linha Verde", "Green Line", "🟢"),
                "linha vermelha": ("Linha Vermelha", "Red Line", "🔴"),
            }

            normalized_result = _normalize_token(cleaned_result)
            outside_regular = bool(
                re.search(
                    r"\b(?:outside normal operating hours|outside regular|fora do horario regular)\b",
                    normalized_result,
                )
            )
            window_match = re.search(r"\((\d{2}:\d{2}-\d{2}:\d{2})\)", cleaned_result)
            regular_window = window_match.group(1) if window_match else "06:30-01:00"
            parsed_lines: List[str] = []
            line_matches = re.findall(
                r"(?P<emoji>[🟡🔵🟢🔴])\s+"
                r"(?P<line>Yellow Line|Blue Line|Green Line|Red Line|Linha Amarela|Linha Azul|Linha Verde|Linha Vermelha)"
                r"(?:\s*\([^\n]*\))?\s*\n\s*(?:[✅⚠️🌙]\s*)?(?P<status>[^\n]+)",
                cleaned_result,
                flags=re.IGNORECASE,
            )

            for emoji, raw_line_name, raw_status in line_matches:
                canonical = line_name_map.get(raw_line_name.lower())
                if not canonical:
                    continue
                pt_name, en_name, fallback_emoji = canonical
                status_lower = raw_status.lower().strip()
                if "no disruption reported" in status_lower:
                    localized_status = (
                        "sem perturbações reportadas; fora do horário regular"
                        if language == "pt" and "outside" in status_lower
                        else "no reported disruption; outside regular hours"
                        if "outside" in status_lower
                        else "sem perturbações reportadas"
                        if language == "pt"
                        else "no reported disruption"
                    )
                elif any(token in status_lower for token in ["normal service", "all lines operating normally", "serviço normal", "circulação normal", "ok"]):
                    localized_status = "circulação normal" if language == "pt" else "normal service"
                elif "unavailable" in status_lower or "indispon" in status_lower:
                    localized_status = "estado em tempo real indisponível" if language == "pt" else "real-time status unavailable"
                else:
                    localized_status = raw_status.strip()

                parsed_lines.append(
                    f"- {emoji or fallback_emoji} **{pt_name if language == 'pt' else en_name}**: {localized_status}"
                )

            if not parsed_lines and re.search(
                r"all lines operating normally|circula[cç][aã]o normal em todas as linhas",
                cleaned_result,
                flags=re.IGNORECASE,
            ):
                if language == "pt":
                    parsed_lines.extend(
                        [
                            "- 🟡 **Linha Amarela**: circulação normal",
                            "- 🔵 **Linha Azul**: circulação normal",
                            "- 🟢 **Linha Verde**: circulação normal",
                            "- 🔴 **Linha Vermelha**: circulação normal",
                        ]
                    )
                else:
                    parsed_lines.extend(
                        [
                            "- 🟡 **Yellow Line**: normal service",
                            "- 🔵 **Blue Line**: normal service",
                            "- 🟢 **Green Line**: normal service",
                            "- 🔴 **Red Line**: normal service",
                        ]
                    )

            if parsed_lines:
                title = "### 🚇 **Estado do Metro de Lisboa**" if language == "pt" else "### 🚇 **Lisbon Metro Status**"
                display_lines = list(parsed_lines)
                if outside_regular:
                    service_line = (
                        "- 🌙 **Serviço ao passageiro:** fora do horário regular "
                        f"({regular_window}); confirma operação especial se vais sair agora"
                        if language == "pt"
                        else "- 🌙 **Passenger service:** outside regular operating "
                        f"hours ({regular_window}); confirm special service if travelling now"
                    )
                    display_lines.insert(0, service_line)

                disrupted_lines = []
                for line in parsed_lines:
                    normalized_line = _normalize_token(line)
                    if (
                        "sem perturbacoes" in normalized_line
                        or "no disruption" in normalized_line
                        or "fora do horario regular" in normalized_line
                        or "outside regular" in normalized_line
                    ):
                        continue
                    if not re.search(
                        r":\s*(?:circulação normal|normal service)\s*$",
                        line,
                        flags=re.IGNORECASE,
                    ):
                        disrupted_lines.append(line)
                normalized_query = _normalize_token(user_message)
                asks_disruption_polarity = bool(
                    re.search(
                        r"\b(?:perturbacao|perturbacoes|incidente|incidentes|avaria|avarias|disruption|disruptions|delay|delays|problem|problems)\b",
                        normalized_query,
                    )
                )
                if disrupted_lines:
                    direct_answer = (
                        "✅ **Resposta direta:** Sim, há perturbações reportadas em pelo menos uma linha."
                        if language == "pt"
                        else "✅ **Direct answer:** Yes, at least one line has a reported disruption."
                    )
                else:
                    if language == "pt":
                        if outside_regular:
                            answer_text = (
                                "Não há perturbações reportadas, mas o Metro está fora do horário regular de passageiros neste momento."
                                if asks_disruption_polarity
                                else "As linhas não têm perturbações reportadas, mas isso não confirma circulação agora porque o Metro está fora do horário regular."
                            )
                        else:
                            answer_text = (
                                "Não, não há perturbações reportadas neste momento nas linhas do Metro de Lisboa."
                                if asks_disruption_polarity
                                else "Sim, as linhas do Metro de Lisboa estão reportadas com circulação normal."
                            )
                        direct_answer = f"✅ **Resposta direta:** {answer_text}"
                    else:
                        if outside_regular:
                            answer_text = (
                                "No disruptions are reported, but Metro is outside regular passenger-service hours right now."
                                if asks_disruption_polarity
                                else "The lines have no reported disruption, but that does not confirm trains are running now because Metro is outside regular hours."
                            )
                        else:
                            answer_text = (
                                "No, there are no reported disruptions on Lisbon Metro lines right now."
                                if asks_disruption_polarity
                                else "Yes, Lisbon Metro lines are currently reported with normal service."
                            )
                        direct_answer = f"✅ **Direct answer:** {answer_text}"
                return "\n".join(
                    [
                        title,
                        "",
                        direct_answer,
                        "",
                        "---",
                        "",
                        *display_lines,
                        "",
                        self._build_transport_source_line(
                            language,
                            ["[*Metro de Lisboa*](https://www.metrolisboa.pt)"],
                        ),
                    ]
                ).strip()

        if tool_name == "carris_get_realtime_vehicles":
            route_short_name = str(
                tool_args.get("route_short_name") or tool_args.get("route_id") or ""
            ).strip().upper()
            timestamp_match = re.search(r"Dados de:\s*(\d{2}:\d{2})(?::\d{2})?", cleaned_result)
            feed_timestamp = timestamp_match.group(1) if timestamp_match else datetime.now().strftime("%H:%M")

            active_entries: List[Tuple[str, Optional[str]]] = []
            current_destination: Optional[str] = None
            for raw_line in cleaned_result.splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    continue

                route_match = re.match(
                    r"^(?P<route>[0-9A-Z]+)\s*->\s*(?P<destination>.+?)\s*\[[^\]]+\]$",
                    stripped,
                )
                if route_match:
                    if route_short_name and route_match.group("route").upper() != route_short_name:
                        current_destination = None
                        continue
                    current_destination = route_match.group("destination").strip()
                    active_entries.append((current_destination, None))
                    continue

                stop_match = re.search(r"Pr[oó]xima paragem:\s*(?P<stop>.+)$", stripped, re.IGNORECASE)
                if stop_match and active_entries and current_destination:
                    destination, _ = active_entries[-1]
                    if destination == current_destination:
                        active_entries[-1] = (destination, stop_match.group("stop").strip())

            if active_entries:
                direction_counts: Dict[str, int] = {}
                sample_stops: Dict[str, str] = {}
                for destination, next_stop in active_entries:
                    direction_counts[destination] = direction_counts.get(destination, 0) + 1
                    if next_stop and destination not in sample_stops:
                        sample_stops[destination] = next_stop

                total_active = len(active_entries)
                title = (
                    f"### 🚋 Estado do {route_short_name} em tempo real"
                    if language == "pt"
                    else f"### 🚋 {route_short_name} live snapshot"
                )
                summary_line = (
                    f"- 📡 **Dados em tempo real:** Carris ativo às {feed_timestamp}."
                    if language == "pt"
                    else f"- 📡 **Real-time data:** Carris active at {feed_timestamp}."
                )
                active_line = (
                    f"- 🚋 **Veículos ativos:** {total_active} elétrico(s) {route_short_name} em circulação."
                    if language == "pt"
                    else f"- 🚋 **Active vehicles:** {total_active} {route_short_name} tram(s) currently in service."
                )
                direction_header = "- ↔️ **Sentidos ativos:**" if language == "pt" else "- ↔️ **Active directions:**"
                limitation_line = (
                    "- ⚠️ Este feed mostra veículos em circulação e próxima paragem, mas não confirma se a linha está a horas, atrasada, ou perturbada."
                    if language == "pt"
                    else "- ⚠️ This feed shows vehicles in service and their next stop, but it does not confirm whether the line is on time, delayed, or disrupted."
                )
                fallback_line = (
                    "- 💡 **Fallback:** sem origem e destino não dá para validar uma alternativa porta-a-porta. Pede-me uma rota com origem/destino para eu calcular a ligação concreta."
                    if language == "pt"
                    else "- 💡 **Fallback:** without an origin and destination I cannot validate a door-to-door alternative. Ask me with both endpoints so I can calculate the concrete route."
                )

                output_lines = [title, "", summary_line, active_line, direction_header]
                for destination, count in sorted(direction_counts.items(), key=lambda item: (-item[1], item[0])):
                    stop_note = sample_stops.get(destination)
                    if language == "pt":
                        bullet = f"    - **{destination}**: {count} veículo(s)"
                        if stop_note:
                            bullet += f" · próxima paragem observada: {stop_note}"
                    else:
                        bullet = f"    - **{destination}**: {count} vehicle(s)"
                        if stop_note:
                            bullet += f" · sample next stop: {stop_note}"
                    output_lines.append(bullet)

                output_lines.extend([limitation_line, fallback_line])
                return "\n".join(output_lines).strip()

        if tool_name in {"get_real_time_bus_positions", "get_bus_realtime_locations"}:
            if tool_name == "get_bus_realtime_locations" and tool_args.get("location") and tool_args.get("line_id"):
                serves_location, route_name = _metropolitana_line_serves_location(
                    str(tool_args.get("line_id")),
                    str(tool_args.get("location")),
                )
                if not serves_location:
                    line_id = str(tool_args.get("line_id")).upper()
                    location = str(tool_args.get("location"))
                    route_note = f" Its published route is **{route_name}**." if route_name else ""
                    return (
                        f"### 🚌 **Carris Metropolitana line check**\n\n"
                        f"- ⚠️ **Line {line_id} does not appear to serve {location}.**{route_note}\n"
                        "- The location in the question is likely mistaken; ask with a stop on that line, or use a line that serves the area."
                    )

            if "### 🚌 **Carris Metropolitana" in cleaned_result:
                localized_result = re.split(
                    r"\n\n⚠️\s+Scope:\s+Carris Metropolitana covers",
                    cleaned_result,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0].strip()
                if language == "pt":
                    localized_result = re.sub(
                        r"###\s+🚌\s+\*\*Carris Metropolitana Line\s+([0-9A-Z]+)\s+-\s+Live Buses\*\*",
                        r"### 🚌 **Carris Metropolitana Linha \1 - Autocarros em tempo real**",
                        localized_result,
                        flags=re.IGNORECASE,
                    )
                    localized_result = re.sub(
                        r"\*\*Short answer:\*\*\s*I found\s+\*\*(\d+)\s+active bus(?:es)?\*\*\s+currently reported on line\s+\*\*([0-9A-Z]+)\*\*\.",
                        r"✅ **Resposta direta:** encontrei **\1 autocarro(s) ativo(s)** atualmente reportados na linha **\2**.",
                        localized_result,
                        flags=re.IGNORECASE,
                    )
                    replacements = {
                        "**Current snapshot**": "### 📡 **Snapshot atual**",
                        "**Live vehicles**": "### 🚌 **Veículos em tempo real**",
                        "**Route:**": "**Trajeto:**",
                        "**Active buses:**": "**Autocarros ativos:**",
                        "**Updated:**": "**Atualizado:**",
                        "📡 Data freshness: live Carris Metropolitana feed snapshot.": "📡 **Dados:** snapshot em tempo real da Carris Metropolitana.",
                        "📡 Data freshness: cached Carris Metropolitana vehicle snapshot": "📡 **Dados:** snapshot em cache da Carris Metropolitana",
                        "**Status:**": "**Estado:**",
                        "**Live position:**": "**Posição em tempo real:**",
                        "[Open map]": "[Abrir mapa]",
                        "**Direction:**": "**Direção:**",
                        "**Speed:**": "**Velocidade:**",
                        "**Next stop:**": "**Próxima paragem:**",
                        "Stopped At": "parado na paragem",
                        "In Transit To": "em circulação para",
                        "💡 **Tip:** If you tell me your exact stop, I can narrow this to the most relevant vehicle and direction.": "💡 **Dica:** se indicares a paragem exata, consigo focar no veículo e sentido mais relevantes.",
                    }
                    for source, target in replacements.items():
                        localized_result = localized_result.replace(source, target)
                    localized_result = re.sub(
                        r"\*\*🚌 Bus\s+([^*]+)\*\*",
                        r"**🚌 Veículo \1**",
                        localized_result,
                    )
                    localized_result = re.sub(
                        r"\b(\d+)\s+active buses\b",
                        r"\1 autocarros ativos",
                        localized_result,
                        flags=re.IGNORECASE,
                    )
                return localized_result

            title = (
                f"### 🚌 Carris Metropolitana live buses near {tool_args.get('location')}"
                if tool_name == "get_real_time_bus_positions" and tool_args.get("location")
                else f"### 🚌 Carris Metropolitana live buses - line {tool_args.get('line_id')}"
                if tool_args.get("line_id")
                else "### 🚌 Carris Metropolitana live buses"
            )
            output_lines = [title, ""]
            vehicle_cards = 0
            inside_vehicle = False

            for raw_line in cleaned_result.splitlines():
                stripped = raw_line.strip()
                if not stripped or set(stripped) <= {"=", "-"}:
                    continue
                if stripped.startswith(("🚌 Real-Time", "🚌 **Real-Time")):
                    continue
                if stripped.startswith("🕐 Updated:"):
                    continue
                if stripped.startswith("📍 Radius:"):
                    output_lines.append(f"- {stripped}")
                    continue
                if stripped.startswith("📊 Active buses:"):
                    output_lines.append(f"- {stripped}")
                    continue
                if stripped.startswith("ℹ️ Broad area fallback:"):
                    output_lines.append(f"- {stripped}")
                    continue
                if stripped.startswith("ℹ️") and "omitted" in stripped.lower():
                    output_lines.append(f"- {stripped}")
                    continue
                if stripped.startswith("📡 Data freshness:"):
                    output_lines.append(f"- {stripped}")
                    continue
                if re.match(r"^-\s*(?:🚌|🛑|🚏|📍)\s*\*\*Line\s+", stripped) or re.match(r"^-\s*\*\*🚌\s*Bus\s+", stripped):
                    vehicle_cards += 1
                    if vehicle_cards > 5:
                        inside_vehicle = False
                        continue
                    output_lines.append("")
                    output_lines.append(stripped)
                    inside_vehicle = True
                    continue
                if inside_vehicle and stripped.startswith("- "):
                    if vehicle_cards <= 5:
                        output_lines.append("    " + stripped)
                    continue
                if stripped.startswith("... and"):
                    output_lines.append(stripped)

            if vehicle_cards > 5:
                output_lines.append(f"... and {vehicle_cards - 5} more shown by the live feed.")

            return "\n".join(output_lines).strip()

        if tool_name == "find_direct_bus_lines":
            origin = str(tool_args.get("origin") or "Origin").strip()
            destination = str(tool_args.get("destination") or "Destination").strip()

            if _tool_result_indicates_no_match(cleaned_result):
                return _format_metropolitana_no_direct_summary(
                    cleaned_result,
                    origin=origin,
                    destination=destination,
                    language=language,
                    build_source_line=self._build_transport_source_line,
                )

            filtered_lines: List[str] = []
            direct_count: int | None = None
            for raw_line in cleaned_result.splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    continue
                lowered = stripped.lower()
                if (
                    (stripped.startswith("⚠️") and "scope" in lowered)
                    or stripped.startswith(("💡 For Lisbon city-only", "📌 **Source:", "📌 **Fonte:"))
                    or stripped.startswith("###")
                    or re.search(r"\*\*(?:Buses|Autocarros):\s*", stripped, flags=re.IGNORECASE)
                ):
                    continue

                direct_count_match = re.search(
                    r"\*\*(\d+)\s+(?:direct\s+line\(s\)\s+found|linhas?\s+diretas?(?:\s+encontradas?)?)\**",
                    stripped,
                    flags=re.IGNORECASE,
                )
                if direct_count_match:
                    direct_count = int(direct_count_match.group(1))
                    continue

                line = raw_line.rstrip()
                if language == "pt":
                    line = re.sub(
                        r"\*\*(\d+)\s+direct\s+line\(s\)\s+found\*\*",
                        r"**\1 linhas diretas**",
                        line,
                        flags=re.IGNORECASE,
                    )
                    line = re.sub(
                        r"^\.\.\.\s+and\s+(\d+)\s+more\s+direct\s+lines:",
                        r"... e mais \1 linhas diretas:",
                        line,
                        flags=re.IGNORECASE,
                    )
                    replacements = {
                        "Buses": "Autocarros",
                        "Terminals": "Terminais",
                        "Path": "Percurso",
                        "Passes through": "Passa por",
                        "How to use it": "Como usar",
                        "check the direction shown at the stop before boarding": "confirma o sentido indicado na paragem antes de embarcar",
                        "Costa Da Caparica": "Costa da Caparica",
                    }
                    for old, new in replacements.items():
                        line = line.replace(old, new)
                else:
                    replacements = {
                        "Autocarros": "Buses",
                        "linha(s) direta(s) encontrada(s)": "direct line(s) found",
                        "Terminais": "Terminals",
                        "Outras linhas": "Other lines",
                        "Como usar": "How to use it",
                        "Procure pelo número da linha": "Look for the line number",
                        "Verifique a direção do autocarro": "Check the bus direction",
                        "Horários e paragens": "Schedules and stops",
                        "na paragem": "at the stop",
                        " (ex: ": " (e.g. ",
                    }
                    for old, new in replacements.items():
                        line = line.replace(old, new)

                filtered_lines.append(line)

            if filtered_lines:
                title = (
                    f"### 🚌 **{origin} → {destination}**"
                    if language == "pt"
                    else f"### 🚌 **{origin} → {destination}**"
                )
                if language == "pt":
                    line_label = "linha direta" if direct_count == 1 else "linhas diretas"
                    direct_answer = (
                        f"✅ **Resposta direta:** sim, encontrei **{direct_count} {line_label}** "
                        "da Carris Metropolitana para esta ligação."
                        if direct_count
                        else "✅ **Resposta direta:** sim, encontrei pelo menos uma linha direta da Carris Metropolitana para esta ligação."
                    )
                else:
                    line_label = "direct line" if direct_count == 1 else "direct lines"
                    direct_answer = (
                        f"✅ **Direct answer:** yes, I found **{direct_count} {line_label}** "
                        "from Carris Metropolitana for this trip."
                        if direct_count
                        else "✅ **Direct answer:** yes, I found at least one direct Carris Metropolitana line for this trip."
                    )
                details_heading = (
                    "#### 🚌 Linhas diretas"
                    if language == "pt"
                    else "#### 🚌 Direct lines"
                )
                return "\n".join(
                    [
                        title,
                        "",
                        direct_answer,
                        "",
                        "---",
                        "",
                        details_heading,
                        "",
                        *filtered_lines,
                        "",
                        self._build_transport_source_line(
                            language,
                            ["[*Carris Metropolitana*](https://www.carrismetropolitana.pt)"],
                        ),
                    ]
                ).strip()

        return cleaned_result

    def _build_explicit_multi_mode_route_response(
        self,
        user_message: str,
        context: str,
        origin: str,
        destination: str,
        language: str,
    ) -> Optional[str]:
        """Build a response when the user explicitly asks for multiple route modes."""
        requested_modes = _requested_route_option_modes(user_message)
        if not {"metro", "bus"}.issubset(requested_modes):
            return None

        preferences = _parse_route_mode_preferences(user_message)
        if preferences["exclude_metro"] or preferences["exclude_bus"]:
            return None

        source_links: List[str] = []
        sections: List[str] = []
        origin_display = _get_transport_display_name(origin)
        destination_display = _get_transport_display_name(destination)
        carris_urban_label = "Carris" if language == "pt" else "Carris Urban"
        prefer_metropolitana = _looks_like_carris_metropolitana_query(
            user_message,
            endpoints=(origin, destination),
        )

        metro_response = _build_deterministic_metro_route_response(
            user_message=user_message,
            context=context,
        )
        if metro_response:
            self._record_tool_call(
                "get_route_between_stations",
                {"origin": origin, "destination": destination},
            )
            source_links.append("[*Metro de Lisboa*](https://www.metrolisboa.pt)")
            metro_lines: List[str] = []
            metro_minutes = self._extract_duration_minutes(metro_response)
            if metro_minutes is not None:
                metro_lines.append(
                    f"- ⏳ **Tempo estimado:** ~{metro_minutes} min"
                    if language == "pt"
                    else f"- ⏳ **Estimated time:** ~{metro_minutes} min"
                )
            metro_route_bullets = self._extract_metro_route_bullets(metro_response)
            if metro_route_bullets:
                for bullet in metro_route_bullets:
                    cleaned_bullet = re.sub(
                        r"^-\s*🚶\s*\*\*Siga a p[eé] para\s+(.+?)\*\*\.?$",
                        r"- 🚶 Caminhada final até **\1**.",
                        bullet,
                        flags=re.IGNORECASE,
                    )
                    cleaned_bullet = re.sub(
                        r"^-\s*🚶\s*\*\*Walk to\s+(.+?)\*\*\.?$",
                        r"- 🚶 Final walk to **\1**.",
                        cleaned_bullet,
                        flags=re.IGNORECASE,
                    )
                    metro_lines.append(cleaned_bullet)
            if not metro_lines:
                cleaned_metro = _strip_transport_source_lines(metro_response)
                cleaned_metro = re.sub(r"^###\s+🚇\s+\*\*[^*]+\*\*\s*", "", cleaned_metro).strip()
                if cleaned_metro:
                    metro_lines.append(cleaned_metro)
            if metro_lines:
                section_title = "**🚇 Opção de metro**" if language == "pt" else "**🚇 Metro option**"
                sections.append(f"{section_title}\n\n" + "\n".join(metro_lines))
        else:
            section_title = "**🚇 Opção de metro**" if language == "pt" else "**🚇 Metro option**"
            limitation = (
                f"- ⚠️ Não consegui confirmar uma rota de metro entre **{origin_display}** e **{destination_display}** com os dados disponíveis."
                if language == "pt"
                else f"- ⚠️ I couldn't confirm a Metro route between **{origin_display}** and **{destination_display}** with the available data."
            )
            sections.append(f"{section_title}\n\n{limitation}")

        urban_tool = self._get_tool_by_name("carris_find_routes_between")
        frequency_tool = self._get_tool_by_name("carris_get_service_frequency")
        urban_result = (
            str(
                self._invoke_tool(
                    urban_tool,
                    {"origin": origin, "destination": destination, "search_radius_km": 0.8},
                    tool_name="carris_find_routes_between",
                )
            ).strip()
            if urban_tool and not prefer_metropolitana
            else ""
        )

        def frequency_lookup(route_short_name: str) -> str:
            if not frequency_tool or not route_short_name:
                return ""
            try:
                return str(
                    self._invoke_tool(
                        frequency_tool,
                        {"route_short_name": route_short_name},
                        tool_name="carris_get_service_frequency",
                    )
                ).strip()
            except Exception:
                return ""

        urban_bus_markdown = _format_carris_mode_section_markdown(
            _extract_carris_mode_section(urban_result, "BUSES"),
            language,
            frequency_lookup=frequency_lookup,
        )
        bus_sections: List[str] = []
        if urban_bus_markdown:
            source_links.append("[*Carris*](https://www.carris.pt)")
            recommended_option = _summarize_recommended_carris_option(
                urban_bus_markdown,
                language,
            )
            bus_body_parts = [part for part in [recommended_option, urban_bus_markdown] if part]
            bus_sections.append("\n\n".join(bus_body_parts))

        if not urban_bus_markdown:
            metropolitan_tool_name = "find_bus_routes" if prefer_metropolitana else "find_direct_bus_lines"
            metropolitan_tool = self._get_tool_by_name(metropolitan_tool_name)
            metropolitan_result = (
                str(
                    self._invoke_tool(
                        metropolitan_tool,
                        {"origin": origin, "destination": destination},
                        tool_name=metropolitan_tool_name,
                    )
                ).strip()
                if metropolitan_tool
                else ""
            )
            metropolitan_block = _localize_metropolitana_direct_bus_block(
                _clean_metropolitana_direct_bus_block(metropolitan_result),
                language,
            )
            if metropolitan_result and not _tool_result_indicates_no_match(metropolitan_result) and metropolitan_block:
                source_links.append("[*Carris Metropolitana*](https://www.carrismetropolitana.pt)")
                bus_sections.append(metropolitan_block)

        bus_section_title = (
            f"**🚌 Opção de autocarro ({carris_urban_label})**"
            if language == "pt" and urban_bus_markdown
            else "**🚌 Opção de autocarro**"
            if language == "pt"
            else f"**🚌 Bus option ({carris_urban_label})**"
            if urban_bus_markdown
            else "**🚌 Bus option**"
        )
        if bus_sections:
            sections.append(f"{bus_section_title}\n\n" + "\n\n".join(bus_sections))
        else:
            source_links.extend(
                [
                    "[*Carris*](https://www.carris.pt)",
                    "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
                ]
            )
            limitation = (
                f"- ⚠️ Não consegui confirmar uma opção de autocarro entre **{origin_display}** e **{destination_display}** nos dados disponíveis dos operadores."
                if language == "pt"
                else f"- ⚠️ I couldn't confirm a bus option between **{origin_display}** and **{destination_display}** in the available operator data."
            )
            sections.append(f"{bus_section_title}\n\n{limitation}")

        metro_confirmed = bool(metro_response)
        bus_confirmed = bool(bus_sections)
        if language == "pt":
            title = f"### 🚇🚌 **{origin_display} → {destination_display}**"
            if metro_confirmed and bus_confirmed:
                direct_line = "✅ **Resposta direta:** encontrei uma opção de metro e uma opção de autocarro com dados dos operadores."
            elif metro_confirmed:
                direct_line = "✅ **Resposta direta:** confirmei a opção de metro; não consegui confirmar uma opção de autocarro com os dados disponíveis."
            elif bus_confirmed:
                direct_line = "✅ **Resposta direta:** confirmei a opção de autocarro; não consegui confirmar uma opção de metro com os dados disponíveis."
            else:
                direct_line = "⚠️ **Resposta direta:** não consegui confirmar opções fiáveis de metro ou autocarro com os dados disponíveis."
        else:
            title = f"### 🚇🚌 **{origin_display} → {destination_display}**"
            if metro_confirmed and bus_confirmed:
                direct_line = "✅ **Direct answer:** I found one Metro option and one bus option using operator data."
            elif metro_confirmed:
                direct_line = "✅ **Direct answer:** I confirmed the Metro option; I couldn't confirm a bus option with the available data."
            elif bus_confirmed:
                direct_line = "✅ **Direct answer:** I confirmed the bus option; I couldn't confirm a Metro option with the available data."
            else:
                direct_line = "⚠️ **Direct answer:** I couldn't confirm reliable Metro or bus options with the available data."

        return "\n".join(
            [
                title,
                "",
                direct_line,
                "",
                "---",
                "",
                "\n\n".join(sections),
                "",
                self._build_transport_source_line(language, source_links),
            ]
        ).strip()

    def _build_mode_constrained_route_response(
        self,
        user_message: str,
        context: str = "",
        language: str = "en",
    ) -> Optional[str]:
        """Builds deterministic responses for route queries with explicit mode constraints."""
        endpoints = _extract_route_endpoints(user_message)
        if not endpoints:
            return None

        preferences = _parse_route_mode_preferences(user_message)
        origin, destination = endpoints
        language = language or infer_response_language(user_query=user_message, default="en")
        carris_urban_label = "Carris" if language == "pt" else "Carris Urban"
        prefer_metropolitana = _looks_like_carris_metropolitana_query(user_message, endpoints=endpoints)
        requested_modes = _requested_route_option_modes(user_message)

        surface_alternative_request = bool(
            preferences.get("alternative_mode_request")
            and requested_modes
            and requested_modes <= {"bus", "tram"}
        )
        if surface_alternative_request:
            urban_tool = self._get_tool_by_name("carris_find_routes_between")
            urban_result = (
                str(
                    self._invoke_tool(
                        urban_tool,
                        {"origin": origin, "destination": destination, "search_radius_km": 0.8},
                        tool_name="carris_find_routes_between",
                    )
                ).strip()
                if urban_tool and not prefer_metropolitana
                else ""
            )
            if urban_result and not _tool_result_indicates_no_match(urban_result):
                return _build_carris_surface_route_response(
                    urban_result,
                    user_message,
                    origin,
                    destination,
                )

            metropolitan_tool_name = "find_bus_routes" if prefer_metropolitana else "find_direct_bus_lines"
            metropolitan_tool = self._get_tool_by_name(metropolitan_tool_name)
            metropolitan_result = (
                str(
                    self._invoke_tool(
                        metropolitan_tool,
                        {"origin": origin, "destination": destination},
                        tool_name=metropolitan_tool_name,
                    )
                ).strip()
                if metropolitan_tool
                else ""
            )
            metropolitan_block = _localize_metropolitana_direct_bus_block(
                _clean_metropolitana_direct_bus_block(metropolitan_result),
                language,
            )
            if (
                metropolitan_result
                and not _tool_result_indicates_no_match(metropolitan_result)
                and metropolitan_block
            ):
                if language == "pt":
                    return "\n".join(
                        [
                            f"### 🚌🚋 **{origin} → {destination}**",
                            "",
                            "✅ **Resposta direta:** encontrei uma opção de autocarro de superfície nos dados disponíveis; não confirmei uma opção de elétrico para esta ligação.",
                            "",
                            "---",
                            "",
                            "### 🚌 **Autocarro**",
                            "",
                            metropolitan_block,
                            "",
                            self._build_transport_source_line(
                                language,
                                ["[*Carris Metropolitana*](https://www.carrismetropolitana.pt)"],
                            ),
                        ]
                    ).strip()
                return "\n".join(
                    [
                        f"### 🚌🚋 **{origin} → {destination}**",
                        "",
                        "✅ **Direct answer:** I found a surface bus option in the available data; I could not confirm a tram option for this trip.",
                        "",
                        "---",
                        "",
                        "### 🚌 **Bus**",
                        "",
                        metropolitan_block,
                        "",
                        self._build_transport_source_line(
                            language,
                            ["[*Carris Metropolitana*](https://www.carrismetropolitana.pt)"],
                        ),
                    ]
                ).strip()

            if language == "pt":
                return (
                    f"### 🚌🚋 **{origin} → {destination}**\n\n"
                    "⚠️ **Resposta direta:** não consegui confirmar opções de autocarro ou elétrico para esta ligação com os dados de superfície disponíveis.\n\n"
                    + self._build_transport_source_line(
                        language,
                        [
                            "[*Carris*](https://www.carris.pt)",
                            "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
                        ],
                    )
                )
            return (
                f"### 🚌🚋 **{origin} → {destination}**\n\n"
                "⚠️ **Direct answer:** I could not confirm bus or tram options for this trip with the available surface-transport data.\n\n"
                + self._build_transport_source_line(
                    language,
                    [
                        "[*Carris*](https://www.carris.pt)",
                        "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
                    ],
                )
            )

        if (
            (preferences["metro_only"] or "metro" in _normalize_token(user_message))
            and _normalize_token(destination) == "madeira"
        ):
            timestamp = datetime.now().strftime("%H:%M")
            if language == "pt":
                return (
                    "### ⚠️ **Destino ambíguo: Madeira**\n\n"
                    "- 🏝️ Se queres dizer **Ilha da Madeira**, não existe ligação por metro urbano a partir de Lisboa; é uma deslocação aérea/marítima fora da rede Metro.\n"
                    "- 🚇 Se queres dizer **Rua Humberto Madeira / Avenida da Ilha da Madeira, em Lisboa**, a estação de metro de referência é **Encarnação** (🔴 Linha Vermelha).\n"
                    "- Para um percurso porta-a-porta, indica o ponto de partida e confirma que te referes à morada em Lisboa.\n\n"
                    f"📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Atualizado:** {timestamp}"
                )
            return (
                "### ⚠️ **Ambiguous destination: Madeira**\n\n"
                "- 🏝️ If you mean **Madeira island**, Lisbon's urban metro cannot get you there; that is outside the Metro network.\n"
                "- 🚇 If you mean **Rua Humberto Madeira / Avenida da Ilha da Madeira in Lisbon**, the reference metro station is **Encarnação** (🔴 Red Line).\n"
                "- For a door-to-door route, provide your starting point and confirm that you mean the Lisbon address.\n\n"
                f"📌 **Source:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | **Updated:** {timestamp}"
            )

        if not any(preferences.values()):
            return None

        if preferences.get("alternative_mode_request"):
            multi_mode_response = self._build_explicit_multi_mode_route_response(
                user_message=user_message,
                context=context,
                origin=origin,
                destination=destination,
                language=language,
            )
            if multi_mode_response:
                return multi_mode_response

        if preferences["metro_only"] or (
            preferences["exclude_bus"]
            and not preferences["tram_only"]
            and not preferences["exclude_metro"]
        ):
            metro_response = _build_deterministic_metro_route_response(
                user_message=user_message,
                context=context,
            )
            if metro_response:
                self._record_tool_call(
                    "get_route_between_stations",
                    {"origin": origin, "destination": destination},
                )
                return metro_response
            if preferences["metro_only"]:
                return _build_mode_unavailable_response(
                    mode="metro",
                    origin=origin,
                    destination=destination,
                    language=language,
                )

        urban_tool = self._get_tool_by_name("carris_find_routes_between")
        frequency_tool = self._get_tool_by_name("carris_get_service_frequency")
        urban_result = (
            str(
                self._invoke_tool(
                    urban_tool,
                    {"origin": origin, "destination": destination, "search_radius_km": 0.8},
                    tool_name="carris_find_routes_between",
                )
            ).strip()
            if urban_tool and not prefer_metropolitana
            else ""
        )

        def frequency_lookup(route_short_name: str) -> str:
            if not frequency_tool or not route_short_name:
                return ""
            try:
                return str(
                    self._invoke_tool(
                        frequency_tool,
                        {"route_short_name": route_short_name},
                        tool_name="carris_get_service_frequency",
                    )
                ).strip()
            except Exception:
                return ""

        if preferences["bus_only"] or preferences["exclude_tram"]:
            urban_bus_block = _extract_carris_mode_section(urban_result, "BUSES")
            urban_tram_block = _extract_carris_mode_section(urban_result, "TRAMS")
            metropolitan_tool_name = "find_bus_routes" if prefer_metropolitana else "find_direct_bus_lines"
            metropolitan_tool = self._get_tool_by_name(metropolitan_tool_name)
            urban_bus_markdown = _format_carris_mode_section_markdown(
                urban_bus_block,
                language,
                frequency_lookup=frequency_lookup,
            )

            sections: List[str] = []
            notes: List[str] = []

            if urban_bus_markdown:
                sections.append(f"#### 🚌 {carris_urban_label}\n\n{urban_bus_markdown}")
            elif _carris_section_has_routes(urban_tram_block):
                notes.append(
                    f"- **{carris_urban_label}:** only tram options were found for this trip, not bus-only ones."
                    if language != "pt"
                    else f"- **{carris_urban_label}:** só apareceram opções de elétrico nesta ligação, não opções apenas de autocarro."
                )
            elif urban_result and not _tool_result_indicates_no_match(urban_result):
                notes.append(
                    f"- **{carris_urban_label}:** no bus-only route could be isolated from the available urban result."
                    if language != "pt"
                    else f"- **{carris_urban_label}:** não foi possível isolar uma rota apenas de autocarro no resultado urbano disponível."
                )

            normalized_request = _normalize_token(user_message)
            wants_metropolitana = prefer_metropolitana or bool(
                re.search(
                    r"\b(?:carris metropolitana|metropolitana|suburbano|suburbana|suburban|aml)\b",
                    normalized_request,
                    flags=re.IGNORECASE,
                )
            )
            urban_has_actionable_bus = bool(
                urban_bus_markdown
                and re.search(
                    r"\*\*(?:Próximas partidas|Next departures|Tempo estimado|Estimated travel time):?\*\*",
                    urban_bus_markdown,
                    flags=re.IGNORECASE,
                )
            )
            should_check_metropolitana = bool(
                metropolitan_tool and (wants_metropolitana or not urban_has_actionable_bus)
            )
            metropolitan_result = (
                str(
                    self._invoke_tool(
                        metropolitan_tool,
                        {"origin": origin, "destination": destination},
                        tool_name=metropolitan_tool_name,
                    )
                ).strip()
                if should_check_metropolitana
                else ""
            )
            metropolitan_block = _localize_metropolitana_direct_bus_block(
                _clean_metropolitana_direct_bus_block(metropolitan_result),
                language,
            )

            if (
                metropolitan_result
                and not _tool_result_indicates_no_match(metropolitan_result)
                and (wants_metropolitana or not urban_has_actionable_bus)
            ):
                if metropolitan_block:
                    if re.match(r"^(?:#{1,6}\s+)?🚌\s+\*\*Carris Metropolitana:", metropolitan_block):
                        sections.append(metropolitan_block)
                    else:
                        sections.append(
                            f"#### 🚌 Carris Metropolitana\n\n{metropolitan_block}"
                        )
            elif not urban_has_actionable_bus:
                notes.append(
                    "- **Carris Metropolitana:** no direct suburban bus line was confirmed for this trip."
                    if language != "pt"
                    else "- **Carris Metropolitana:** não foi confirmada nenhuma linha suburbana direta para esta ligação."
                )

            if not sections:
                if language == "pt":
                    message = (
                        f"❌ Não consegui confirmar uma rota apenas de autocarro entre {origin} e {destination} com os dados disponíveis da {carris_urban_label} e da Carris Metropolitana."
                    )
                else:
                    message = (
                        f"❌ I couldn't confirm a bus-only route between {origin} and {destination} with the available {carris_urban_label} and Carris Metropolitana data."
                    )
                if notes:
                    notes_title = "#### ℹ️ Coverage notes" if language != "pt" else "#### ℹ️ Notas de cobertura"
                    message += "\n\n" + notes_title + "\n" + "\n".join(notes)
                message += "\n\n" + self._build_transport_source_line(
                    language,
                    [
                        "[*Carris*](https://www.carris.pt)",
                        "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
                    ],
                )
                return message

            intro = (
                f"### 🚌 Bus-only options for {origin} → {destination}"
                if language != "pt"
                else f"### 🚌 Opções apenas de autocarro para {origin} → {destination}"
            )
            direct_line = (
                f"✅ **Direct answer:** I found bus-only options between **{origin}** and **{destination}** in the available operator data."
                if language != "pt"
                else f"✅ **Resposta direta:** encontrei opções apenas de autocarro entre **{origin}** e **{destination}** nos dados disponíveis dos operadores."
            )
            recommended_option = _summarize_recommended_carris_option(
                urban_bus_markdown,
                language,
            )
            response_parts = [intro, "", direct_line]
            if recommended_option:
                response_parts.extend(["", recommended_option])
            response_parts.extend(["", "---", "", "\n\n".join(sections)])
            source_links = []
            if urban_bus_markdown:
                source_links.append("[*Carris*](https://www.carris.pt)")
            if metropolitan_result and not _tool_result_indicates_no_match(metropolitan_result):
                source_links.append("[*Carris Metropolitana*](https://www.carrismetropolitana.pt)")
            if not source_links:
                source_links = [
                    "[*Carris*](https://www.carris.pt)",
                    "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
                ]
            if notes:
                notes_title = "#### ℹ️ Coverage notes" if language != "pt" else "#### ℹ️ Notas de cobertura"
                response_parts.extend(["", notes_title, *notes])
            response_parts.extend(
                [
                    "",
                    self._build_transport_source_line(
                        language,
                        source_links,
                    ),
                ]
            )
            return "\n".join(response_parts).strip()

        if preferences["exclude_metro"]:
            urban_bus_block = _extract_carris_mode_section(urban_result, "BUSES")
            urban_tram_block = _extract_carris_mode_section(urban_result, "TRAMS")
            metropolitan_tool_name = "find_bus_routes" if prefer_metropolitana else "find_direct_bus_lines"
            metropolitan_tool = self._get_tool_by_name(metropolitan_tool_name)
            metropolitan_result = (
                str(
                    self._invoke_tool(
                        metropolitan_tool,
                        {"origin": origin, "destination": destination},
                        tool_name=metropolitan_tool_name,
                    )
                ).strip()
                if metropolitan_tool
                else ""
            )

            urban_bus_markdown = _format_carris_mode_section_markdown(
                urban_bus_block,
                language,
                frequency_lookup=frequency_lookup,
            )
            urban_tram_markdown = _format_carris_mode_section_markdown(
                urban_tram_block,
                language,
                frequency_lookup=frequency_lookup,
            )
            metropolitan_block = _localize_metropolitana_direct_bus_block(
                _clean_metropolitana_direct_bus_block(metropolitan_result),
                language,
            )

            sections: List[str] = []
            if urban_bus_markdown:
                sections.append(f"#### 🚌 {carris_urban_label}\n\n{urban_bus_markdown}")
            if urban_tram_markdown:
                sections.append(f"#### 🚋 {carris_urban_label}\n\n{urban_tram_markdown}")
            if metropolitan_result and not _tool_result_indicates_no_match(metropolitan_result) and metropolitan_block:
                if re.match(r"^(?:#{1,6}\s+)?🚌\s+\*\*Carris Metropolitana:", metropolitan_block):
                    sections.append(metropolitan_block)
                else:
                    sections.append(f"#### 🚌 Carris Metropolitana\n\n{metropolitan_block}")

            if not sections:
                if language == "pt":
                    return (
                        f"❌ Não consegui confirmar uma rota sem metro entre {origin} e {destination} com os dados de superfície disponíveis.\n\n"
                        + self._build_transport_source_line(
                            language,
                            [
                                "[*Carris*](https://www.carris.pt)",
                                "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
                            ],
                        )
                    )
                return (
                    f"❌ I couldn't confirm a non-metro surface route between {origin} and {destination} with the available data.\n\n"
                    + self._build_transport_source_line(
                        language,
                        [
                            "[*Carris*](https://www.carris.pt)",
                            "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
                        ],
                    )
                )

            intro = (
                f"### 🚌🚋 Surface options without metro for {origin} → {destination}"
                if language != "pt"
                else f"### 🚌🚋 Opções de superfície sem metro para {origin} → {destination}"
            )
            direct_line = (
                f"✅ **Direct answer:** I found surface options without Metro between **{origin}** and **{destination}** in the available operator data."
                if language != "pt"
                else f"✅ **Resposta direta:** encontrei opções de superfície sem Metro entre **{origin}** e **{destination}** nos dados disponíveis dos operadores."
            )
            return "\n".join(
                [
                    intro,
                    "",
                    direct_line,
                    "",
                    "---",
                    "",
                    "\n\n".join(sections),
                    "",
                    self._build_transport_source_line(
                        language,
                        [
                            "[*Carris*](https://www.carris.pt)",
                            "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
                        ],
                    ),
                ]
            ).strip()

        if preferences["tram_only"] or preferences["exclude_bus"]:
            urban_tram_block = _extract_carris_mode_section(urban_result, "TRAMS")
            if _carris_section_has_routes(urban_tram_block):
                urban_tram_markdown = _format_carris_mode_section_markdown(
                    urban_tram_block,
                    language,
                    frequency_lookup=frequency_lookup,
                )
                intro = (
                    f"### 🚋 Tram-only options for {origin} → {destination}"
                    if language != "pt"
                    else f"### 🚋 Opções apenas de elétrico para {origin} → {destination}"
                )
                return "\n".join(
                    [
                        intro,
                        "",
                        f"#### 🚋 {carris_urban_label}",
                        "",
                        urban_tram_markdown or urban_tram_block,
                        "",
                        self._build_transport_source_line(
                            language,
                            ["[*Carris*](https://www.carris.pt)"],
                        ),
                    ]
                ).strip()

            if language == "pt":
                return (
                    f"❌ Não consegui confirmar uma rota sem autocarro entre {origin} e {destination} com os dados disponíveis.\n\n"
                    f"📌 **Fonte:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | [*Carris*](https://www.carris.pt) | **Atualizado:** {datetime.now().strftime('%H:%M')}"
                )
            return (
                f"❌ I couldn't confirm a non-bus route between {origin} and {destination} with the available data.\n\n"
                f"📌 **Source:** [*Metro de Lisboa*](https://www.metrolisboa.pt) | [*Carris*](https://www.carris.pt) | **Updated:** {datetime.now().strftime('%H:%M')}"
            )

        return None

    def _resolve_deterministic_response(
        self,
        user_message: str,
        context: str = "",
        language: Optional[str] = None,
    ) -> Optional[str]:
        """Resolves deterministic direct-response fast paths shared by invoke() and build_subgraph()."""
        resolved_language = language or infer_response_language(user_query=user_message, default="en")
        unsupported_mode_response = _build_unsupported_transport_scope_response(
            user_message=user_message,
            language=resolved_language,
        )
        if unsupported_mode_response:
            return self._finalize_transport_response(
                unsupported_mode_response,
                user_message=user_message,
                language=resolved_language,
            )

        deterministic_tool_msg = _build_deterministic_transport_tool_call(user_message)
        deterministic_tool_name = (
            deterministic_tool_msg.tool_calls[0].get("name")
            if deterministic_tool_msg and deterministic_tool_msg.tool_calls
            else None
        )
        non_route_tool_names = {
            "get_carris_metropolitana_alerts",
            "get_transport_summary",
            "get_metro_status",
            "get_all_metro_stations",
            "get_metro_line_wait_times",
            "get_bus_realtime_locations",
            "get_bus_next_departures",
            "search_carris_metropolitana_lines",
            "get_carris_metropolitana_stop_info",
        }
        endpoints = [] if deterministic_tool_name in non_route_tool_names else _extract_route_endpoints(user_message)
        quality_sensitive_route = bool(
            endpoints
            and _query_has_route_quality_preferences(user_message)
            and not _query_has_route_mode_constraints(user_message)
        )
        ambiguity_preamble = ""
        if endpoints:
            try:
                from tools.location_resolver import build_location_ambiguity_preamble

                ambiguity_preamble = build_location_ambiguity_preamble(
                    endpoints[0],
                    endpoints[1],
                    language=resolved_language,
                )
            except Exception:
                ambiguity_preamble = ""

        def with_ambiguity_preamble(response: str) -> str:
            """Prepend bare-location ambiguity context when a fallback rewrites route output."""
            if not ambiguity_preamble:
                return response
            if "Ambiguidade" in response or "Ambiguity" in response:
                return response
            return f"{ambiguity_preamble}\n\n{response}".strip()

        def finalize_direct_route_response(response: str) -> str:
            """Finalize a generic route answer and record the consulted route layers."""
            if endpoints:
                self._record_tool_call(
                    "get_route_between_stations",
                    {"origin": endpoints[0], "destination": endpoints[1]},
                )
                if _route_response_uses_carris_urban(response):
                    self._record_tool_call(
                        "carris_find_routes_between",
                        {"origin": endpoints[0], "destination": endpoints[1]},
                    )
            finalized_response = self._finalize_transport_response(
                response,
                user_message=user_message,
                language=resolved_language,
                ensure_wait_times=True,
            )
            return with_ambiguity_preamble(finalized_response)

        if ambiguity_preamble and not _ambiguity_preamble_is_no_clear_match(ambiguity_preamble):
            return self._finalize_transport_response(
                ambiguity_preamble,
                user_message=user_message,
                language=resolved_language,
            )

        if endpoints and route_mentions_outside_aml(user_message):
            return self._finalize_transport_response(
                build_geographic_out_of_scope_response(user_message, language=resolved_language),
                user_message=user_message,
                language=resolved_language,
            )

        if deterministic_tool_name in non_route_tool_names:
            deterministic_tool_response = self._invoke_deterministic_tool_call(
                user_message,
                resolved_language,
            )
            if deterministic_tool_response:
                return deterministic_tool_response

        setubal_scope_response = _build_lisbon_setubal_scope_response(
            user_message=user_message,
            language=resolved_language,
        )
        if setubal_scope_response:
            return self._finalize_transport_response(
                setubal_scope_response,
                user_message=user_message,
                language=resolved_language,
            )

        if not quality_sensitive_route:
            cp_multileg_response = self._build_cp_multileg_response(
                user_message=user_message,
                language=resolved_language,
            )
            if cp_multileg_response:
                return self._finalize_transport_response(
                    cp_multileg_response,
                    user_message=user_message,
                    language=resolved_language,
                )

        early_cp_tool_spec = _build_cp_tool_spec(user_message)
        explicit_train_context = bool(
            re.search(r"\b(?:cp|comboio|comboios|train|trains)\b", user_message, flags=re.IGNORECASE)
            and not re.search(
                r"\b(?:sem|evitar|evito|n[aã]o\s+quero|without|avoid|no)\s+"
                r"(?:cp|comboio|comboios|train|trains)\b",
                user_message,
                flags=re.IGNORECASE,
            )
        )
        requested_route_modes = _requested_route_option_modes(user_message) if endpoints else set()
        surface_only_route_modes = bool(requested_route_modes) and requested_route_modes <= {"bus", "tram"}
        if endpoints and surface_only_route_modes:
            constrained_route_response = self._build_mode_constrained_route_response(
                user_message=user_message,
                context=context,
                language=resolved_language,
            )
            if constrained_route_response:
                finalized_response = self._finalize_transport_response(
                    constrained_route_response,
                    user_message=user_message,
                    language=resolved_language,
                )
                return with_ambiguity_preamble(finalized_response)

        if (
            endpoints
            and _is_generic_public_transport_route_query(user_message)
            and not surface_only_route_modes
            and not quality_sensitive_route
            and not (early_cp_tool_spec and explicit_train_context)
        ):
            early_direct_route_response = _build_deterministic_route_tool_response(user_message)
            if early_direct_route_response:
                return finalize_direct_route_response(early_direct_route_response)

        nearest_cp_destination_response = None
        if not quality_sensitive_route and not (early_cp_tool_spec and explicit_train_context):
            nearest_cp_destination_response = self._build_nearest_cp_destination_response(
                user_message=user_message,
                language=resolved_language,
            )
        if nearest_cp_destination_response:
            return self._finalize_transport_response(
                nearest_cp_destination_response,
                user_message=user_message,
                language=resolved_language,
            )

        madeira_station_response = self._build_madeira_nearest_metro_response(
            user_message=user_message,
            language=resolved_language,
        )
        if madeira_station_response:
            return self._finalize_transport_response(
                madeira_station_response,
                user_message=user_message,
                language=resolved_language,
            )

        non_station_metro_response = self._build_non_station_nearest_metro_response(
            user_message=user_message,
            language=resolved_language,
        )
        if non_station_metro_response:
            return self._finalize_transport_response(
                non_station_metro_response,
                user_message=user_message,
                language=resolved_language,
            )

        comparison_response = self._build_mode_comparison_response(
            user_message=user_message,
            context=context,
            language=resolved_language,
        )
        if comparison_response:
            finalized_response = self._finalize_transport_response(
                comparison_response,
                user_message=user_message,
                language=resolved_language,
            )
            return with_ambiguity_preamble(finalized_response)

        destination_overview_response = _build_destination_only_transport_overview_response(
            user_message=user_message,
            context=context,
        )
        if destination_overview_response:
            finalized_response = self._finalize_transport_response(
                destination_overview_response,
                user_message=user_message,
                language=resolved_language,
            )
            return with_ambiguity_preamble(finalized_response)

        cp_tool_spec = _build_cp_tool_spec(user_message)
        explicit_train_context = bool(
            re.search(r"\b(?:cp|comboio|comboios|train|trains)\b", user_message, flags=re.IGNORECASE)
            and not re.search(
                r"\b(?:sem|evitar|evito|n[aã]o\s+quero|without|avoid|no)\s+"
                r"(?:cp|comboio|comboios|train|trains)\b",
                user_message,
                flags=re.IGNORECASE,
            )
        )
        if cp_tool_spec and explicit_train_context:
            tool_name = str(cp_tool_spec.get("name") or "")
            tool_args = dict(cp_tool_spec.get("args") or {})
            tool = self._get_tool_by_name(tool_name)
            if tool:
                result = self._invoke_tool(tool, tool_args, tool_name=tool_name)
                if _tool_result_indicates_no_match(str(result)) and re.search(
                    r"\b(?:sem|evitar|evito|n[aã]o\s+quero|without|avoid|no)\s+"
                    r"(?:autocarro|autocarros|bus|buses)\b",
                    user_message,
                    flags=re.IGNORECASE,
                ):
                    train_metro_response = self._build_train_metro_no_bus_response(
                        user_message=user_message,
                        language=resolved_language,
                    )
                    if train_metro_response:
                        return self._finalize_transport_response(
                            train_metro_response,
                            user_message=user_message,
                            language=resolved_language,
                        )
                formatted_result = self._format_deterministic_tool_result(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result=str(result).strip(),
                    language=resolved_language,
                    user_message=user_message,
                )
                finalized_response = self._finalize_transport_response(
                    formatted_result,
                    user_message=user_message,
                    language=resolved_language,
                )
                return with_ambiguity_preamble(finalized_response)

        constrained_route_response = self._build_mode_constrained_route_response(
            user_message=user_message,
            context=context,
            language=resolved_language,
        )
        if constrained_route_response:
            finalized_response = self._finalize_transport_response(
                constrained_route_response,
                user_message=user_message,
                language=resolved_language,
            )
            return with_ambiguity_preamble(finalized_response)

        metro_wait_response = _build_deterministic_metro_wait_response(
            user_message=user_message,
            context=context,
        )
        if metro_wait_response:
            line_wait_line_id = (
                _extract_metro_line_id(user_message)
                if _query_requests_metro_line_wait_times(user_message)
                else None
            )
            if line_wait_line_id:
                self._record_tool_call("get_metro_line_wait_times", {"line": line_wait_line_id})
            elif (
                _query_requests_all_metro_lines_wait_times(user_message)
                or _query_requests_top_metro_wait_times(user_message)
            ):
                for _line_id in ("amarela", "azul", "verde", "vermelha"):
                    self._record_tool_call("get_metro_line_wait_times", {"line": _line_id})
            else:
                metro_wait_request = _parse_metro_wait_request(user_message)
                if metro_wait_request:
                    requested_line_id = _extract_metro_line_id(user_message)
                    if requested_line_id and not metro_wait_request.get("direction"):
                        self._record_tool_call("get_metro_line_wait_times", {"line": requested_line_id})
                    else:
                        wait_args = {"station": metro_wait_request.get("station")}
                        if metro_wait_request.get("direction"):
                            wait_args["direction"] = metro_wait_request.get("direction")
                        self._record_tool_call("get_metro_wait_time", wait_args)
            return self._finalize_transport_response(
                metro_wait_response,
                user_message=user_message,
                language=resolved_language,
            )

        if _query_requests_future_cp_schedule(user_message):
            return self._finalize_transport_response(
                _build_future_cp_schedule_limit_response(user_message, resolved_language),
                user_message=user_message,
                language=resolved_language,
            )

        if _query_requests_broad_carris_catalog(user_message):
            return self._finalize_transport_response(
                _build_broad_carris_catalog_limit_response(resolved_language),
                user_message=user_message,
                language=resolved_language,
            )

        deterministic_response = None
        cp_tool_spec = _build_cp_tool_spec(user_message)
        explicit_surface_mode = bool(
            re.search(
                r"\b(carris|autocarro|autocarros|bus|buses|tram|trams|el[eé]trico|eletrico)\b",
                user_message,
                flags=re.IGNORECASE,
            )
        )
        if (
            not ambiguity_preamble
            and not cp_tool_spec
            and not explicit_surface_mode
            and not _is_generic_public_transport_route_query(user_message)
            and not quality_sensitive_route
        ):
            deterministic_response = _build_deterministic_metro_route_response(
                user_message=user_message,
                context=context,
            )
        if deterministic_response:
            if endpoints:
                self._record_tool_call(
                    "get_route_between_stations",
                    {"origin": endpoints[0], "destination": endpoints[1]},
                )
            return self._finalize_transport_response(
                deterministic_response,
                user_message=user_message,
                language=resolved_language,
            )

        carris_stop_response = _build_deterministic_carris_stop_response(user_message, resolved_language)
        if carris_stop_response:
            carris_stop_request = _parse_carris_line_stop_query(user_message)
            if carris_stop_request:
                kind = carris_stop_request.get("kind")
                line = carris_stop_request.get("line")
                stop_reference = carris_stop_request.get("stop_id") or carris_stop_request.get("stop_name")
                if kind == "arrivals":
                    self._record_tool_call(
                        "carris_get_arrivals",
                        {"stop": stop_reference, "limit": 8},
                    )
                elif kind == "departures":
                    self._record_tool_call(
                        "carris_get_next_departures",
                        {"route_short_name": line, "stop": stop_reference},
                    )
                elif kind == "eta":
                    self._record_tool_call(
                        "carris_vehicle_eta",
                        {"route_short_name": line, "stop": stop_reference},
                    )
            return self._finalize_transport_response(
                carris_stop_response,
                user_message=user_message,
                language=resolved_language,
            )

        multipart_metropolitana_response = self._build_metropolitana_multipart_response(
            user_message,
            resolved_language,
        )
        if multipart_metropolitana_response:
            return self._finalize_transport_response(
                multipart_metropolitana_response,
                user_message=user_message,
                language=resolved_language,
            )

        carris_metropolitana_tool_spec = _build_carris_metropolitana_tool_spec(user_message)

        if not cp_tool_spec and not carris_metropolitana_tool_spec:
            direct_route_response = None if quality_sensitive_route else _build_deterministic_route_tool_response(user_message)
            if direct_route_response:
                return finalize_direct_route_response(direct_route_response)

        direct_tool_response = self._run_direct_tool_fallback(user_message)
        if direct_tool_response:
            finalized_response = self._finalize_transport_response(
                direct_tool_response,
                user_message=user_message,
                language=resolved_language,
            )
            return with_ambiguity_preamble(finalized_response)

        return None

    def _invoke_deterministic_tool_call(
        self,
        user_message: str,
        language: Optional[str] = None,
    ) -> Optional[str]:
        """Invokes a deterministic single-tool fast path and finalizes the result for the user."""
        resolved_language = language or infer_response_language(user_query=user_message, default="en")
        if _query_requests_future_cp_schedule(user_message):
            return self._finalize_transport_response(
                _build_future_cp_schedule_limit_response(user_message, resolved_language),
                user_message=user_message,
                language=resolved_language,
            )
        if _query_requests_broad_carris_catalog(user_message):
            return self._finalize_transport_response(
                _build_broad_carris_catalog_limit_response(resolved_language),
                user_message=user_message,
                language=resolved_language,
            )

        tool_call_msg = _build_deterministic_transport_tool_call(user_message)
        if not tool_call_msg or not tool_call_msg.tool_calls:
            return None

        tool_call = tool_call_msg.tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool = self._get_tool_by_name(tool_name)
        if not tool:
            return None

        invoke_args = dict(tool_args)
        if tool_name == "get_bus_realtime_locations" and "location" in invoke_args:
            invoke_args = {"line_id": invoke_args.get("line_id")}

        result = self._invoke_tool(tool, invoke_args, tool_name=tool_name)
        formatted_result = self._format_deterministic_tool_result(
            tool_name=tool_name,
            tool_args=tool_args,
            result=str(result).strip(),
            language=resolved_language,
            user_message=user_message,
        )
        finalized_response = self._finalize_transport_response(
            formatted_result,
            user_message=user_message,
            language=resolved_language,
            ensure_wait_times=tool_name == "get_route_between_stations",
        )
        return self._prepend_location_ambiguity(
            finalized_response,
            tool_args,
            resolved_language,
        )

    def _build_subgraph_deterministic_tool_call(
        self,
        user_message: str,
    ) -> Optional[AIMessage]:
        """Builds deterministic tool-call messages for subgraph execution and coverage tracking."""
        tool_call_msg = _build_deterministic_transport_tool_call(user_message)
        if tool_call_msg:
            return tool_call_msg

        query_lower = user_message.lower()
        if (
            not _query_has_wait_departure_intent(user_message)
            and (
                _query_is_aggregate_transport_status(user_message)
                or re.search(r"\b(summary|overview|all transport|transport summary|transport overview|across)\b", query_lower)
            )
            and re.search(r"\b(transport|metro|bus|buses|train|trains|comboio|comboios|autocarro|autocarros)\b", query_lower)
        ):
            language = infer_response_language(user_query=user_message, default="en")
            return _build_tool_call("get_transport_summary", {"language": language})

        endpoints = _extract_route_endpoints(user_message)
        if endpoints:
            if route_mentions_outside_aml(user_message):
                return None
            if (
                _query_has_route_quality_preferences(user_message)
                and not _query_has_route_mode_constraints(user_message)
            ):
                return None
            return _build_tool_call(
                "get_route_between_stations",
                {"origin": endpoints[0], "destination": endpoints[1]},
            )

        return None

    @staticmethod
    def _is_deterministic_subgraph_tool_result(messages: List) -> bool:
        """Returns True when the current ToolMessage came from an auto-generated fast-path tool call."""
        if len(messages) < 2 or not isinstance(messages[-1], ToolMessage):
            return False

        previous_message = messages[-2]
        if not isinstance(previous_message, AIMessage) or not getattr(previous_message, "tool_calls", None):
            return False

        tool_call_id = previous_message.tool_calls[0].get("id", "")
        return isinstance(tool_call_id, str) and tool_call_id.startswith("auto_")

    @staticmethod
    def _build_language_instruction(language: str) -> str:
        """Builds a compact language instruction for LLM-backed subgraph steps."""
        return (
            "Respond ENTIRELY in Portuguese (PT-PT)."
            if language == "pt"
            else "Respond ENTIRELY in English."
        )

    def _ensure_subgraph_messages(
        self,
        messages: List,
        language: str,
    ) -> List:
        """Ensures transport subgraph LLM steps receive system and language instructions."""
        updated_messages = list(messages)
        if not updated_messages or not isinstance(updated_messages[0], SystemMessage):
            updated_messages = [SystemMessage(content=self._get_runtime_system_prompt(language))] + updated_messages

        if not any(
            isinstance(message, SystemMessage)
            and "Respond ENTIRELY" in str(message.content)
            for message in updated_messages[:3]
        ):
            updated_messages = [
                updated_messages[0],
                SystemMessage(content=self._build_language_instruction(language)),
                *updated_messages[1:],
            ]

        return updated_messages

    def _ensure_realtime_wait_times(self, user_message: str, response: str) -> str:
        """Guarantees real-time wait times for metro route responses."""
        if not response:
            return response

        if _is_future_transport_planning_query(user_message):
            return response

        user_lower = user_message.lower()
        response_lower = response.lower()
        if "metro" not in user_lower and "trajeto de metro" not in response_lower:
            return response

        endpoints = _extract_route_endpoints(user_message)
        if not endpoints:
            return response

        from tools.transport_api import get_route_between_stations

        try:
            route_result = str(
                get_route_between_stations.invoke(
                    {"origin": endpoints[0], "destination": endpoints[1]}
                )
            )
        except Exception:
            route_result = ""

        targets = _parse_wait_targets_from_route(route_result)
        if not targets:
            targets = _parse_wait_targets_from_response(response)
        if not targets:
            return response

        language = infer_response_language(user_query=user_message, default="en")
        realtime_lines = _build_metro_wait_lines(targets, language=language)

        return _upsert_realtime_wait_section(response, realtime_lines, language=language)

    @staticmethod
    def _infer_route_operators_from_response(response: str) -> List[str]:
        """Infer checked transport operators from a route response body."""
        raw = response or ""
        normalized = _normalize_token(raw)
        operators: List[str] = []

        def add(operator: str) -> None:
            if operator not in operators:
                operators.append(operator)

        if any(
            marker in normalized
            for marker in [
                "metro de lisboa",
                "metro route",
                "trajeto de metro",
                "your metro route",
                "linha vermelha",
                "linha amarela",
                "linha azul",
                "linha verde",
                "red line",
                "yellow line",
                "blue line",
                "green line",
            ]
        ):
            add("metro")

        if "carris metropolitana" in normalized:
            add("carris_metropolitana")
        elif "carris urban" in normalized or "carris urbana" in normalized:
            add("carris")

        has_carris_direct = bool(
            re.search(
                r"(?im)^(?:####?\s*)?(?:🚌\s*)?(?:autocarros|buses|el[eé]tricos|trams)\b|"
                r"\b(?:direct routes found|rotas diretas encontradas)\b|"
                r"\b(?:linha|line)\s+\d{2,3}[A-Z]?\b|"
                r"\*\*\d{2,3}[A-Z]?\*\*",
                raw,
            )
        )
        if has_carris_direct and "carris_metropolitana" not in normalized:
            add("carris")

        if re.search(r"\b(?:comboio via|train via|cp suburban|cp suburbana|linha de cascais|linha de sintra)\b", normalized):
            add("cp")

        return operators

    def _finalize_transport_response(
        self,
        response: str,
        *,
        user_message: str,
        language: str,
        ensure_wait_times: bool = False,
    ) -> str:
        """Finalize a transport response and rebuild its footer from recorded operators."""
        formatted_response = response
        if ensure_wait_times:
            formatted_response = self._ensure_realtime_wait_times(user_message, formatted_response)

        if _is_preformatted_metro_route_response(formatted_response):
            finalized = formatted_response.strip()
        else:
            finalized = finalize_worker_response(
                formatted_response,
                agent_name="transport",
                user_query=user_message,
                language=language,
            )
        finalized = preserve_contextual_destination_name(finalized, user_message, language)

        tool_names = [call.get("tool_name") for call in self.get_tool_calls_log()]
        operators_used = operators_from_tool_names(tool_names)
        tool_name_set = {str(name or "") for name in tool_names}
        if "get_route_between_stations" in tool_name_set:
            for operator in self._infer_route_operators_from_response(finalized):
                if operator not in operators_used:
                    operators_used.append(operator)
        if operators_used:
            rebuilt = rebuild_transport_source_line(finalized, operators_used, language=language)
            if rebuilt != finalized:
                finalized = rebuilt
            elif not has_source_line(finalized):
                source_links = {
                    "metro": "[*Metro de Lisboa*](https://www.metrolisboa.pt)",
                    "carris": "[*Carris*](https://www.carris.pt)",
                    "carris_metropolitana": "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
                    "cp": "[*CP*](https://www.cp.pt)",
                }
                finalized = (
                    f"{finalized}\n\n"
                    f"{self._build_transport_source_line(language, [source_links[op] for op in operators_used if op in source_links])}"
                ).strip()
        else:
            # Transport source footers must reflect checked operators. If a
            # deterministic limitation path did not invoke a transport tool,
            # keep the answer but remove inherited/static operator citations.
            finalized = _strip_transport_source_lines(finalized)

        return finalized

    @traceable(name="transport_agent", run_type="chain", tags=["sub-agent", "transport"])
    def invoke(
        self, user_message: str, context: str = "", verbose: bool = False
    ) -> str:
        """
        Processes a transport-related query.

        Args:
            user_message: The user's query.
            context: Additional context from other agents (optional).
            verbose: Whether involved tool calls should be printed.

        Returns:
            str: Transport information response.
        """
        self.reset_llm_usage_tracking()

        # Extract explicit language preference from context if provided
        import re
        language_match = re.search(r"User language:\s*(en|pt)", context, re.IGNORECASE)
        if language_match:
            language = language_match.group(1).lower()
        else:
            language = infer_response_language(user_query=user_message, default="en")
        effective_user_message = self._rewrite_follow_up_transport_query(user_message, language, context)
        language_instruction = (
            "Respond ENTIRELY in Portuguese (PT-PT)."
            if language == "pt"
            else "Respond ENTIRELY in English."
        )

        system_prompt = self._get_runtime_system_prompt(language)
        messages = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=language_instruction),
        ]

        if context:
            messages.append(
                SystemMessage(content=f"Context from other agents:\n{context}")
            )

        messages.append(HumanMessage(content=effective_user_message))

        # Skip tool enforcement for greetings/thanks
        is_greeting = any(
            w in effective_user_message.lower()
            for w in ["hello", "thanks", "obrigado", "tchau", "olá", "bom dia"]
        )

        if not is_greeting:
            deterministic_response = self._resolve_deterministic_response(
                user_message=effective_user_message,
                context=context,
                language=language,
            )
            if deterministic_response:
                self._remember_transport_context(effective_user_message)
                return deterministic_response

            deterministic_tool_response = self._invoke_deterministic_tool_call(
                user_message=effective_user_message,
                language=language,
            )
            if deterministic_tool_response:
                self._remember_transport_context(effective_user_message)
                return deterministic_tool_response

        response = self.execute_react_loop(
            messages=messages,
            verbose=verbose,
            max_iterations=6,
            tool_enforcement_msg="" if is_greeting else (
                "You MUST use a tool (like get_metro_status or get_route_between_stations) "
                "to get real data. Do NOT answer from your knowledge base. Call the tool now."
            ),
        )

        self._remember_transport_context(effective_user_message)
        return self._finalize_transport_response(
            response,
            user_message=effective_user_message,
            language=language,
            ensure_wait_times=True,
        )

    def build_subgraph(self) -> "CompiledStateGraph":
        """
        Builds a LangGraph subgraph for this agent.

        Returns:
            CompiledStateGraph: Compiled subgraph for transport queries.
        """

        def agent_node(state: AgentState) -> dict:
            """Transport agent decision node."""
            messages = list(state["messages"])

            user_message = None
            for message in reversed(messages):
                if isinstance(message, HumanMessage) and message.content:
                    user_message = str(message.content)
                    break

            language = infer_response_language(user_query=user_message or "", default="en")

            last_message = messages[-1] if messages else None
            if isinstance(last_message, ToolMessage):
                if self._is_deterministic_subgraph_tool_result(messages):
                    previous_message = messages[-2]
                    tool_call = getattr(previous_message, "tool_calls", [{}])[0]
                    tool_name = str(tool_call.get("name") or "")
                    tool_args = tool_call.get("args", {})
                    if tool_name and not any(call.get("tool_name") == tool_name for call in self.get_tool_calls_log()):
                        self._record_tool_call(tool_name, dict(tool_args))
                    formatted_response = self._format_deterministic_tool_result(
                        tool_name=tool_name,
                        tool_args=dict(tool_args),
                        result=str(last_message.content).strip(),
                        language=language,
                        user_message=user_message or "",
                    )
                    finalized_response = self._finalize_transport_response(
                        formatted_response,
                        user_message=user_message or "",
                        language=language,
                        ensure_wait_times=tool_name == "get_route_between_stations",
                    )
                    finalized_response = self._prepend_location_ambiguity(
                        finalized_response,
                        tool_args,
                        language,
                    )
                    return {"messages": [AIMessage(content=finalized_response)]}

                response = self._safe_llm_invoke(
                    self.llm_with_tools,
                    self._ensure_subgraph_messages(messages, language),
                )
                return {"messages": [response]}

            if user_message:
                deterministic_response = self._resolve_deterministic_response(
                    user_message=user_message,
                    context="",
                    language=language,
                )
                if deterministic_response:
                    return {"messages": [AIMessage(content=deterministic_response)]}

                deterministic_tool_call = self._build_subgraph_deterministic_tool_call(user_message)
                if deterministic_tool_call:
                    return {"messages": [deterministic_tool_call]}

            response = self._safe_llm_invoke(
                self.llm_with_tools,
                self._ensure_subgraph_messages(messages, language),
            )
            return {"messages": [response]}

        def should_continue(state: AgentState) -> str:
            """Determines next step."""
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)

        def fallback_tools_node(state: AgentState) -> dict:
            """Executes tool calls without LangGraph's ToolNode when tests inject mock-like tools."""
            last_message = state["messages"][-1]
            tool_messages: List[ToolMessage] = []

            for tool_call in getattr(last_message, "tool_calls", []) or []:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", f"auto_{uuid.uuid4().hex}")
                tool = self._get_tool_by_name(tool_name)

                if tool is None:
                    result = f"Tool '{tool_name}' not found."
                else:
                    try:
                        result = self._invoke_tool(tool, tool_args, tool_name=tool_name)
                    except Exception as exc:
                        result = f"Error executing {tool_name}: {exc}"

                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )

            return {"messages": tool_messages}

        try:
            tools_node = ToolNode(self.tools)
        except ValueError:
            tools_node = fallback_tools_node

        workflow.add_node("tools", tools_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Transport Agent Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    import os

    counters = {"passed": 0, "failed": 0}

    def _check(condition: bool, label: str) -> None:
        if condition:
            counters["passed"] += 1
            print(f"   \033[1;32m✅ PASS\033[0m: {label}")
        else:
            counters["failed"] += 1
            print(f"   \033[1;31m❌ FAIL\033[0m: {label}")

    deterministic_cases = {
        "Is the metro working?": "get_metro_status",
        "When are the next trains from Entrecampos?": "get_train_schedule",
        "How do I get from Rossio to Sintra by train?": "plan_train_trip",
        "Show real-time Carris Metropolitana buses near Almada": "get_real_time_bus_positions",
        "What are the direct Carris Metropolitana buses from Oeiras to Amadora?": "find_direct_bus_lines",
    }

    print("\n\033[1m📝 Offline deterministic smoke checks:\033[0m")
    for query, expected_tool in deterministic_cases.items():
        deterministic_call = _build_deterministic_transport_tool_call(query)
        resolved_tool = None
        if deterministic_call and deterministic_call.tool_calls:
            resolved_tool = deterministic_call.tool_calls[0].get("name")
            print(f"   🔧 {query} -> {resolved_tool}")
        _check(resolved_tool == expected_tool, f"Deterministic routing selects {expected_tool} for '{query}'")

    parsed_carris_query = _parse_carris_line_stop_query("What are the next departures for 732 at Rossio?")
    _check(
        isinstance(parsed_carris_query, dict)
        and parsed_carris_query.get("kind") == "departures"
        and parsed_carris_query.get("line") == "732",
        "Carris stop/line parser extracts departures queries correctly",
    )
    _check(
        _is_future_transport_planning_query("Como vou amanhã do Rossio ao Aeroporto de metro?") is True,
        "Future transport planning detector recognises tomorrow-style queries",
    )

    formatting_agent = TransportAgent.__new__(TransportAgent)
    formatted_cm_output = formatting_agent._format_deterministic_tool_result(
        tool_name="find_direct_bus_lines",
        tool_args={"origin": "Oeiras", "destination": "Amadora"},
        result=(
            "🚌 **Buses: Oeiras → Amadora**\n\n"
            "✅ **19 direct line(s) found:**\n\n"
            "**1. 🚍 Linha 1501**\n"
            " 📍 **Terminals**: Reboleira (Estação) | Circular via Alfragide\n"
            "💡 **How to use it:**\n"
            " - Look for the line number\n"
            "⚠️ **Scope**: raw wrapper that should disappear"
        ),
        language="en",
    )
    _check(formatted_cm_output.startswith("### 🚌 Direct Carris Metropolitana lines"), "Carris Metropolitana formatter adds a clean title")
    _check("Scope" not in formatted_cm_output, "Carris Metropolitana formatter strips raw scope lines")
    _check("How to use it" in formatted_cm_output, "Carris Metropolitana formatter preserves helpful usage hints")

    future_wait_response = _build_future_metro_wait_limit_response("Saldanha", "Odivelas", "pt")
    _check("tempo real" in future_wait_response.lower(), "Future metro wait response explains the real-time limitation")

    if os.getenv("LISBOA_RUN_LIVE_TRANSPORT_TESTS") == "1":
        print("\n\033[1m🌐 Optional live transport smoke:\033[0m")
        try:
            live_agent = TransportAgent()
            for query in [
                "Is the metro working?",
                "What are the direct Carris Metropolitana buses from Oeiras to Amadora?",
            ]:
                live_response = live_agent.invoke(query)
                print(f"\n{query}\n{'-' * len(query)}")
                print(live_response[:1200])
                _check(bool(live_response.strip()), f"Live transport smoke returned content for '{query}'")
        except Exception as exc:
            _check(False, f"Live transport smoke failed: {exc}")
    else:
        print("\n   ℹ️ Live transport smoke skipped. Set LISBOA_RUN_LIVE_TRANSPORT_TESTS=1 to enable it.")

    print(f"\n\033[1mSummary:\033[0m Passed={counters['passed']} Failed={counters['failed']}")
    if counters["failed"]:
        raise SystemExit(1)
    print("\n\033[1;32m✅ Transport agent smoke test passed!\033[0m")
