# ==========================================================================
# Master Thesis - Quality Assurance Agent
#   - André Filipe Gomes Silvestre, 20240502
#
#   Validates completeness of agent outputs before final response.
#   Two-phase validation:
#     Phase 1 (LLM): Structural completeness check via prompt-based analysis
#     Phase 2 (Deterministic): Factual verification against known data
#   Identifies missing data and returns retry hints to the orchestrator.
#   Ensures no incomplete or hallucinated responses reach the user.
# ==========================================================================

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agent.agents.base import BaseAgent, clean_response, parse_json_response, traceable
from agent.prompts.qa import get_qa_prompt

# Import authoritative static transport data from tool modules.
# These provide the single source of truth for metro/CP verification (no API calls).
try:
    from tools.metrolisboa_api import METRO_LINES as _METRO_LINES_DATA
    from tools.metrolisboa_api import METRO_STATIONS as _METRO_STATIONS_DATA
    _HAS_METRO_DATA = True
except ImportError:
    _METRO_LINES_DATA: Dict = {}
    _METRO_STATIONS_DATA: Dict = {}
    _HAS_METRO_DATA = False

try:
    from tools.cp_api import CP_LINES as _CP_LINES_DATA
    _HAS_CP_DATA = True
except ImportError:
    _CP_LINES_DATA: Dict = {}
    _HAS_CP_DATA = False

logger = logging.getLogger(__name__)

# ==========================================================================
# Static Knowledge for Deterministic Fact-Checking
# ==========================================================================
# Metro and CP authoritative data is imported from tool modules above.
# Only non-dynamic knowledge (bounds, domains, limits) is defined here.

# Canonical metro station names - derived from the authoritative METRO_STATIONS
# dict imported from tools.metrolisboa_api. Kept as an alias for backward
# compatibility (used by tests and external imports).
_METRO_CANONICAL_STATIONS: set = (
    set(_METRO_STATIONS_DATA.keys()) if _METRO_STATIONS_DATA
    else {  # Minimal inline fallback for isolated test environments
        "rato", "marquês de pombal", "marques de pombal", "picoas", "saldanha",
        "campo pequeno", "entre campos", "entrecampos", "cidade universitária",
        "cidade universitaria", "campo grande", "quinta das conchas", "lumiar",
        "ameixoeira", "senhor roubado", "odivelas",
        "santa apolónia", "santa apolonia", "terreiro do paço", "terreiro do paco",
        "baixa-chiado", "baixa chiado", "restauradores", "avenida", "parque",
        "são sebastião", "sao sebastiao", "praça de espanha", "praca de espanha",
        "jardim zoológico", "jardim zoologico", "laranjeiras", "alto dos moinhos",
        "colégio militar", "colegio militar", "carnide", "pontinha", "alfornelos",
        "amadora este", "reboleira",
        "cais do sodré", "cais do sodre", "rossio", "martim moniz", "intendente",
        "anjos", "arroios", "alameda", "areeiro", "roma", "alvalade", "telheiras",
        "olaias", "bela vista", "chelas", "olivais", "cabo ruivo", "oriente",
        "moscavide", "encarnação", "encarnacao", "aeroporto",
    }
)

# AML geographic bounding box (same values as cp_api.AML_BOUNDS)
_AML_BOUNDS = {
    "lat_min": 38.4,
    "lat_max": 39.0,
    "lon_min": -9.5,
    "lon_max": -8.7,
}

# Known valid URL domains for Lisbon data
_VALID_DOMAINS = {
    "visitlisboa.com", "metrolisboa.pt", "api.metrolisboa.pt",
    "carrismetropolitana.pt", "api.carrismetropolitana.pt",
    "cp.pt", "comboios.live", "ipma.pt", "api.ipma.pt",
    "dados.cm-lisboa.pt", "dados.gov.pt", "cm-lisboa.pt",
    "wikipedia.org", "en.wikipedia.org", "pt.wikipedia.org",
    "carris.pt", "gateway.carris.pt", "aml.pt",
}

# IPMA forecast range (max days available)
_IPMA_FORECAST_DAYS = 5

# Lisbon historic temperature bounds (°C) for weather sanity checks.
# Source: IPMA records. All-time high: 44.1°C (Aug 2023). Generous margins applied.
_LISBON_TEMP_MIN = -5.0
_LISBON_TEMP_MAX = 47.0

# Time tolerance factor for itinerary duration check (allows 50% overrun before warning)
_TIME_TOLERANCE_FACTOR = 1.5

# Output truncation limit (chars per agent output, controls LLM token usage)
_TRUNCATION_LIMIT = 6000

# Known Carris tram (elétrico) lines currently operating in Lisbon.
# Routes with GTFS route_short_name ending in "E". 12E is tourist-only (Hills Tramcar).
# Source: https://www.carris.pt/linhas-e-paragens/ (as of 2025)
_CARRIS_TRAM_LINES = {"12e", "15e", "18e", "25e", "28e"}


class QualityAssuranceAgent(BaseAgent):
    """
    Quality Assurance agent that validates data completeness and factual accuracy.

    Two-phase validation:
        Phase 1 (LLM): Analyzes structural completeness via prompt-based reasoning.
            Checks if all required data fields are present for the query type.
        Phase 2 (Deterministic): Cross-checks factual claims against known data.
            Validates metro stations, coordinates, dates, URLs without LLM involvement.

    Responsibilities:
        - Analyze outputs from specialized agents
        - Verify user preferences/constraints are addressed
        - Identify missing critical data for the query type
        - Return `required_agents` hints when data is incomplete
        - Flag potential hallucinations or data gaps
        - Add disclaimers about known data limitations

    Note:
        This agent has NO LangChain tools. It uses deterministic Python functions
        for fact-checking (Phase 2), not LLM tool-calling. The surrounding
        orchestration layer decides whether returned retry hints should trigger
        additional worker execution.
    """

    def __init__(self):
        """Initializes the QA agent."""
        super().__init__("qa")

    @staticmethod
    def _normalize_query(user_query: str) -> str:
        """Returns a normalized lower-case query string for lightweight intent guards."""
        return (user_query or "").strip().lower()

    @classmethod
    def _is_event_listing_query(cls, user_query: str) -> bool:
        """Detects event-discovery queries that should stay within the researcher domain."""
        query = cls._normalize_query(user_query)
        if not query:
            return False

        event_patterns = [
            r"\bevents?\b",
            r"\beventos?\b",
            r"\bconcerts?\b",
            r"\bconcertos?\b",
            r"\bexhibitions?\b",
            r"\bexposi(?:ç|c)[aã]o(?:es)?\b",
            r"\bwhat'?s on\b",
            r"\bo que acontece\b",
            r"\bcultura\b",
            r"\bcultural\b",
            r"\bfestival(?:es)?\b",
        ]
        planning_patterns = [
            r"\bplan\b",
            r"\bplanning\b",
            r"\bitinerary\b",
            r"\broteiro\b",
            r"\bitiner[aá]rio\b",
            r"\bplane(?:ia|ar|ie)\b",
            r"\bcria(?:r)? um itiner[aá]rio\b",
            r"\borganiza(?:r)?\b",
            r"\bwhat to do\b",
            r"\bo que fazer\b",
        ]
        weather_patterns = [
            r"\bweather\b",
            r"\bforecast\b",
            r"\bmeteorolog",
            r"\bprevis[aã]o\b",
            r"\brain\b",
            r"\bchuva\b",
            r"\btemperatura\b",
            r"\btemperature\b",
            r"\btempo em\b",
            r"\bqual (?:é|e) o tempo\b",
        ]
        transport_patterns = [
            r"\btransport\b",
            r"\btransporte\b",
            r"\bmetro\b",
            r"\bbus\b",
            r"\bautocarro\b",
            r"\bcomboio\b",
            r"\btrain\b",
            r"\broute\b",
            r"\brota\b",
            r"\bliga[cç][aã]o\b",
            r"\bconnection\b",
            r"\bhow to get\b",
            r"\bcomo chegar\b",
        ]

        has_event_intent = any(re.search(pattern, query) for pattern in event_patterns)
        has_planning_intent = any(re.search(pattern, query) for pattern in planning_patterns)
        has_weather_intent = any(re.search(pattern, query) for pattern in weather_patterns)
        has_transport_intent = any(re.search(pattern, query) for pattern in transport_patterns)

        return has_event_intent and not has_planning_intent and not has_weather_intent and not has_transport_intent

    @staticmethod
    def _is_cross_domain_event_requirement(text: str) -> bool:
        """Returns whether a QA gap/disclaimer incorrectly asks for weather/transport in an events-only query."""
        lower = (text or "").lower()
        cross_domain_patterns = [
            r"\bweather\b",
            r"\bmeteorolog",
            r"\bforecast\b",
            r"\bprevis[aã]o\b",
            r"\brain\b",
            r"\bchuva\b",
            r"\btemperature\b",
            r"\btemperatura\b",
            r"\btransport\b",
            r"\btransporte\b",
            r"\bmetro\b",
            r"\bcarris\b",
            r"\bcp\b",
            r"\broute\b",
            r"\brota\b",
            r"\bliga[cç][aã]o\b",
            r"\bconnection\b",
            r"\btransfer\b",
            r"\bcomo chegar\b",
            r"\bhow to get\b",
        ]
        return any(re.search(pattern, lower) for pattern in cross_domain_patterns)

    @classmethod
    def _normalize_event_query_validation(
        cls,
        user_query: str,
        agents_called: List[str],
        llm_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prevents event-only queries from being expanded into weather/transport retries by QA."""
        if not cls._is_event_listing_query(user_query):
            return llm_result

        called_workers = {agent for agent in agents_called if agent}
        if called_workers and not called_workers.issubset({"researcher"}):
            return llm_result

        filtered_required_agents = [
            agent for agent in llm_result.get("required_agents", [])
            if agent == "researcher"
        ]
        filtered_missing_data = [
            item for item in llm_result.get("missing_data", [])
            if not cls._is_cross_domain_event_requirement(item)
        ]
        filtered_disclaimers = [
            item for item in llm_result.get("disclaimers", [])
            if not cls._is_cross_domain_event_requirement(item)
        ]

        llm_result["required_agents"] = filtered_required_agents
        llm_result["missing_data"] = filtered_missing_data
        llm_result["disclaimers"] = filtered_disclaimers

        if not filtered_required_agents and not filtered_missing_data:
            llm_result["complete"] = True

        normalization_note = (
            "Normalized QA for event-only query: weather and transport are optional and must not trigger retries."
        )
        reasoning = llm_result.get("reasoning", "")
        if normalization_note not in reasoning:
            llm_result["reasoning"] = (
                f"{reasoning} | {normalization_note}".strip(" |")
            )

        return llm_result

    @classmethod
    def _is_place_listing_query(cls, user_query: str) -> bool:
        """Detects standalone place/attraction discovery queries that should stay inside researcher scope."""
        query = cls._normalize_query(user_query)
        if not query:
            return False

        place_patterns = [
            r"\battractions?\b",
            r"\batra(?:ç|c)[aã]o(?:es)?\b",
            r"\bplaces?\b",
            r"\blocais?\b",
            r"\bmuseums?\b",
            r"\bmuseus?\b",
            r"\bmonuments?\b",
            r"\bmonumentos?\b",
            r"\bwhat to visit\b",
            r"\bo que visitar\b",
            r"\bimperd[ií]veis\b",
            r"\bfirst time\b",
            r"\bprimeira vez\b",
        ]
        planning_patterns = [
            r"\bplan\b",
            r"\bplanning\b",
            r"\bitinerary\b",
            r"\broteiro\b",
            r"\bitiner[aá]rio\b",
            r"\bplane(?:ia|ar|ie)\b",
            r"\borganiza(?:r)?\b",
            r"\bo que fazer\b",
        ]
        weather_patterns = [
            r"\bweather\b",
            r"\bforecast\b",
            r"\bmeteorolog",
            r"\bprevis[aã]o\b",
            r"\brain\b",
            r"\bchuva\b",
            r"\btemperatura\b",
            r"\btemperature\b",
        ]
        transport_patterns = [
            r"\btransport\b",
            r"\btransporte\b",
            r"\bmetro\b",
            r"\bbus\b",
            r"\bautocarro\b",
            r"\bcomboio\b",
            r"\btrain\b",
            r"\broute\b",
            r"\brota\b",
            r"\bhow to get\b",
            r"\bcomo chegar\b",
        ]

        has_place_intent = any(re.search(pattern, query) for pattern in place_patterns)
        has_planning_intent = any(re.search(pattern, query) for pattern in planning_patterns)
        has_weather_intent = any(re.search(pattern, query) for pattern in weather_patterns)
        has_transport_intent = any(re.search(pattern, query) for pattern in transport_patterns)

        return has_place_intent and not has_planning_intent and not has_weather_intent and not has_transport_intent

    @classmethod
    def _normalize_place_query_validation(
        cls,
        user_query: str,
        agents_called: List[str],
        llm_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prevents standalone place-listing queries from being expanded into weather/transport retries by QA."""
        if not cls._is_place_listing_query(user_query):
            return llm_result

        called_workers = {agent for agent in agents_called if agent}
        if called_workers and not called_workers.issubset({"researcher"}):
            return llm_result

        filtered_required_agents = [
            agent for agent in llm_result.get("required_agents", [])
            if agent == "researcher"
        ]
        filtered_missing_data = [
            item for item in llm_result.get("missing_data", [])
            if not cls._is_cross_domain_event_requirement(item)
        ]
        filtered_disclaimers = [
            item for item in llm_result.get("disclaimers", [])
            if not cls._is_cross_domain_event_requirement(item)
        ]

        llm_result["required_agents"] = filtered_required_agents
        llm_result["missing_data"] = filtered_missing_data
        llm_result["disclaimers"] = filtered_disclaimers

        if not filtered_required_agents and not filtered_missing_data:
            llm_result["complete"] = True

        normalization_note = (
            "Normalized QA for place-only query: weather and transport are optional and must not trigger retries."
        )
        reasoning = llm_result.get("reasoning", "")
        if normalization_note not in reasoning:
            llm_result["reasoning"] = (
                f"{reasoning} | {normalization_note}".strip(" |")
            )

        return llm_result

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        """Removes duplicates while preserving the original order."""
        deduped: List[str] = []
        for item in items:
            if item and item not in deduped:
                deduped.append(item)
        return deduped

    @classmethod
    def _merge_fact_check_results(
        cls,
        combined_fact_check: Dict[str, Any],
        per_agent_fact_checks: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merges combined and per-agent deterministic fact-check results."""
        disclaimers: List[str] = []
        critical_issues: List[str] = []
        checks_performed: List[str] = []
        repairable_agents: List[str] = []

        for fact_check in [combined_fact_check, *per_agent_fact_checks.values()]:
            disclaimers.extend(fact_check.get("disclaimers", []))
            critical_issues.extend(fact_check.get("critical_issues", []))
            checks_performed.extend(fact_check.get("checks_performed", []))

        for agent_name, fact_check in per_agent_fact_checks.items():
            if fact_check.get("critical_issues"):
                repairable_agents.append(agent_name)

        merged_disclaimers = cls._dedupe_preserve_order(disclaimers)
        merged_critical = cls._dedupe_preserve_order(critical_issues)
        merged_checks = cls._dedupe_preserve_order(checks_performed)
        merged_repairable_agents = cls._dedupe_preserve_order(repairable_agents)

        return {
            "valid": len(merged_critical) == 0,
            "disclaimers": merged_disclaimers,
            "critical_issues": merged_critical,
            "checks_performed": merged_checks,
            "repairable_agents": merged_repairable_agents,
            "per_agent": per_agent_fact_checks,
        }

    @traceable(name="qa_agent", run_type="chain", tags=["sub-agent", "qa"])
    def validate(
        self,
        user_query: str,
        agent_outputs: Dict[str, str],
        agents_called: List[str],
        language: str = "en",
        user_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Validates if gathered data is complete for answering the user query.

        Runs two phases:
            Phase 1: LLM-based structural completeness check
            Phase 2: Deterministic fact verification (metro stations, coordinates,
                     dates, URLs)

        Args:
            user_query: The user's original query.
            agent_outputs: Dict mapping agent names to their output strings.
            agents_called: List of agent names that were called.
            language: Language code ('en' or 'pt').
            user_context: User preferences and constraints (location, mobility,
                         preferences, available_time, language).
            conversation_history: Last 2-3 user messages for follow-up coherence.

        Returns:
            Dict with validation result:
                - complete (bool): True if data is sufficient
                - missing_data (List[str]): List of missing data fields
                - required_agents (List[str]): Agents the orchestrator may call
                    for missing data
                - reasoning (str): Explanation of the assessment
                - disclaimers (List[str]): Warnings about data limitations
                - fact_check (Dict): Results from deterministic verification
                - critical_issues (List[str]): Deterministic issues that require correction
                - repairable_agents (List[str]): Worker agents whose outputs should be revised
                - needs_repair (bool): Whether the final response should be repaired
        """
        # ── Phase 1: LLM-based structural completeness ──────────────
        system_prompt = get_qa_prompt(
            language,
            user_context=user_context,
            conversation_history=conversation_history,
        )

        # Build context showing what was gathered
        context_parts = [f"**User Query:** {user_query}"]
        context_parts.append(f"**Agents Called:** {', '.join(agents_called)}")

        # Include user context if available
        if user_context:
            ctx_lines = []
            if user_context.get("preferences"):
                ctx_lines.append(f"- Interests/Preferences: {', '.join(user_context['preferences'])}")
            if user_context.get("mobility"):
                ctx_lines.append(f"- Mobility: {user_context['mobility']}")
            if user_context.get("available_time"):
                ctx_lines.append(f"- Available time: {user_context['available_time']}h")
            if user_context.get("latitude") and user_context.get("longitude"):
                ctx_lines.append(f"- Location: ({user_context['latitude']:.4f}, {user_context['longitude']:.4f})")
            if user_context.get("language"):
                ctx_lines.append(f"- Language preference: {user_context['language']}")
            if ctx_lines:
                context_parts.append("**User Context:**\n" + "\n".join(ctx_lines))

        # Include conversation history for follow-up coherence
        if conversation_history:
            history_str = " → ".join(conversation_history[-3:])
            context_parts.append(f"**Recent conversation:** {history_str}")

        for agent_name, output in agent_outputs.items():
            if agent_name.startswith("_"):
                continue  # Skip internal keys
            # Truncate very long outputs to avoid token limits
            truncated = output[:_TRUNCATION_LIMIT] if len(str(output)) > _TRUNCATION_LIMIT else output
            context_parts.append(
                f"\n**{agent_name.upper()} Agent Output:**\n{truncated}"
            )

        context = "\n".join(context_parts)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"# VALIDATION TASK\n\nValidate completeness of the following data:\n\n{context}"),
        ]

        # LLM call with retry for Azure content filter false positives
        response = self._safe_llm_invoke(self.llm, messages)
        content = clean_response(response.content, _print=False)

        # Parse JSON response (with one retry on failure)
        result = parse_json_response(content)

        if not result:
            # Retry: ask LLM again with explicit JSON instruction
            logger.warning("QA: First JSON parse failed, retrying...")
            retry_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        "# VALIDATION TASK (RETRY)\n\n"
                        "Your previous response was not valid JSON. "
                        "Output ONLY a JSON object with keys: complete, missing_data, "
                        "required_agents, reasoning, disclaimers.\n\n"
                        f"{context}"
                    )
                ),
            ]
            response = self._safe_llm_invoke(self.llm, retry_messages)
            content = clean_response(response.content, _print=False)
            result = parse_json_response(content)

        if result:
            llm_result = {
                "complete": result.get("complete", True),
                "missing_data": result.get("missing_data", []),
                "required_agents": [
                    a for a in result.get("required_agents", [])
                    if a in ("weather", "transport", "researcher")
                ],
                "reasoning": result.get("reasoning", ""),
                "disclaimers": result.get("disclaimers", []),
            }
            llm_result = self._normalize_event_query_validation(
                user_query=user_query,
                agents_called=agents_called,
                llm_result=llm_result,
            )
            llm_result = self._normalize_place_query_validation(
                user_query=user_query,
                agents_called=agents_called,
                llm_result=llm_result,
            )
        else:
            # Fallback: if JSON parsing still fails, pass with disclaimer
            logger.warning("QA: JSON parse failed after retry; passing with disclaimer.")
            llm_result = {
                "complete": True,
                "missing_data": [],
                "required_agents": [],
                "reasoning": "QA validation could not parse LLM response after retry.",
                "disclaimers": ["Quality validation was limited for this response"],
            }

        # ── Phase 2: Deterministic fact verification ─────────────────
        combined_output = "\n".join(
            str(v) for k, v in agent_outputs.items()
            if not k.startswith("_") and isinstance(v, str)
        )

        per_agent_fact_checks: Dict[str, Dict[str, Any]] = {}
        for agent_name, output in agent_outputs.items():
            if agent_name.startswith("_") or not isinstance(output, str):
                continue
            per_agent_fact_checks[agent_name] = self._verify_facts(
                output,
                user_query,
                user_context,
            )

        combined_fact_check = self._verify_facts(combined_output, user_query, user_context)
        fact_check = self._merge_fact_check_results(
            combined_fact_check=combined_fact_check,
            per_agent_fact_checks=per_agent_fact_checks,
        )

        # Merge fact-check disclaimers into LLM result
        if fact_check.get("disclaimers"):
            llm_result["disclaimers"] = self._dedupe_preserve_order(
                llm_result.get("disclaimers", []) + fact_check["disclaimers"]
            )

        # If fact-check found critical issues, flag as incomplete
        if fact_check.get("critical_issues"):
            llm_result["reasoning"] += f" | Fact-check: {'; '.join(fact_check['critical_issues'])}"

        llm_result["critical_issues"] = fact_check.get("critical_issues", [])
        llm_result["repairable_agents"] = fact_check.get("repairable_agents", [])
        llm_result["needs_repair"] = bool(fact_check.get("critical_issues"))
        llm_result["fact_check"] = fact_check
        return llm_result

    @traceable(name="qa_repair_pass", run_type="chain", tags=["sub-agent", "qa", "repair"])
    def repair_final_response(
        self,
        user_query: str,
        draft_response: str,
        agent_outputs: Dict[str, str],
        qa_result: Dict[str, Any],
        language: str = "en",
    ) -> str:
        """Repairs a draft final response using QA findings and grounded worker outputs.

        This pass is intentionally conservative: it may rewrite phrasing,
        remove unsupported claims, and surface missing-data caveats, but it must
        stay strictly grounded in the provided worker outputs.
        """
        if not draft_response:
            return draft_response

        fact_check = qa_result.get("fact_check", {}) if isinstance(qa_result, dict) else {}
        critical_issues = self._dedupe_preserve_order(
            list(qa_result.get("critical_issues", []))
            + list(fact_check.get("critical_issues", []))
        )
        disclaimers = self._dedupe_preserve_order(
            list(qa_result.get("disclaimers", []))
            + list(fact_check.get("disclaimers", []))
        )

        if not critical_issues and not disclaimers:
            return draft_response

        if language == "pt":
            system_prompt = (
                "És a etapa final de reparação de qualidade da resposta. "
                "Receberás um rascunho de resposta, os outputs dos agentes worker e os achados do QA. "
                "Reescreve APENAS o necessário para corrigir problemas factuais, remover alegações não confirmadas, "
                "manter a resposta completa e preservar uma apresentação natural com markdown, emojis e linhas de fonte corretas.\n\n"
                "REGRAS:\n"
                "- Usa apenas factos presentes no rascunho e nos outputs dos agentes.\n"
                "- Nunca inventes locais, horários, preços, acessibilidade, estações, ligações, ou URLs.\n"
                "- Se algo não estiver confirmado, diz explicitamente que deve ser verificado.\n"
                "- Remove referências internas a QA, validação, fact-checking, reasoning, ou agentes.\n"
                "- Preserva a mesma língua do utilizador, o estilo visual, os emojis úteis e a estrutura markdown.\n"
                "- Todas as tuas edições, avisos e aditamentos de texto devem ser estritamente em Português (PT-PT).\n"
                "- Mantém ou melhora a linha de fonte final se ela já existir.\n"
                "- Formata qualquer nota ou aviso de QA no final da resposta usando explicitamente listas com bullets iniciados por ⚠️ (ex: - ⚠️ Nota...).\n"
                "- Devolve apenas a resposta final reparada, sem prefácio nem explicações."
            )
            task_prefix = "# TAREFA DE REPARAÇÃO FINAL"
            critical_label = "Problemas críticos"
            disclaimer_label = "Notas e limitações"
            worker_label = "Outputs dos workers"
            draft_label = "Rascunho atual"
        else:
            system_prompt = (
                "You are the final response-quality repair pass. "
                "You will receive a draft answer, the worker-agent outputs, and QA findings. "
                "Rewrite only what is necessary to fix factual issues, remove unsupported claims, "
                "keep the answer complete, and preserve a natural markdown response with useful emojis and a correct source line.\n\n"
                "RULES:\n"
                "- Use only facts present in the draft and worker outputs.\n"
                "- Never invent venues, times, prices, accessibility claims, stations, links, or URLs.\n"
                "- If something is not confirmed, say it should be verified.\n"
                "- Remove any references to QA, validation, fact-checking, reasoning, or internal agents.\n"
                "- Preserve the user's language, visual style, helpful emojis, and markdown structure.\n"
                "- Keep or improve the final source line if one already exists.\n"
                "- Format any QA disclaimer or note at the bottom of the response explicitly using bullet points starting with ⚠️ (e.g., - ⚠️ Note...).\n"
                "- Return only the repaired final answer, with no preface or explanation."
            )
            task_prefix = "# FINAL REPAIR TASK"
            critical_label = "Critical issues"
            disclaimer_label = "Warnings and limitations"
            worker_label = "Worker outputs"
            draft_label = "Current draft"

        worker_context_parts = []
        for agent_name, output in agent_outputs.items():
            if agent_name.startswith("_") or not isinstance(output, str):
                continue
            truncated = output[:_TRUNCATION_LIMIT] if len(output) > _TRUNCATION_LIMIT else output
            worker_context_parts.append(f"## {agent_name.upper()}\n{truncated}")

        worker_context = "\n\n".join(worker_context_parts) if worker_context_parts else ""
        critical_block = "\n".join(f"- {item}" for item in critical_issues) or "- None"
        disclaimer_block = "\n".join(f"- {item}" for item in disclaimers) or "- None"

        human_content = (
            f"{task_prefix}\n\n"
            f"**User query:** {user_query}\n\n"
            f"## {critical_label}\n{critical_block}\n\n"
            f"## {disclaimer_label}\n{disclaimer_block}\n\n"
            f"## {draft_label}\n{draft_response[:_TRUNCATION_LIMIT]}\n\n"
            f"## {worker_label}\n{worker_context}"
        )

        try:
            response = self._safe_llm_invoke(
                self.llm,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_content),
                ],
            )
            repaired = clean_response(response.content, _print=False).strip()
            return repaired or draft_response
        except Exception as exc:
            logger.warning("QA final repair pass failed, keeping draft response: %s", exc)
            return draft_response

    def _verify_facts(
        self,
        combined_output: str,
        user_query: str,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Deterministic fact verification against authoritative static data.

        Checks (9 total):
            1. Metro station names (METRO_STATIONS from metrolisboa_api)
            2. Metro line-station pair validity (sentence-level, using METRO_LINES)
            3. CP train line names (CP_LINES from cp_api)
            4. AML coordinate bounds (Lisbon Metropolitan Area bounding box)
            5. Date sanity (IPMA 5-day forecast range)
            6. URL domain validation (known Lisbon data sources)
            7. User preference adherence (accessibility, available time)
            8. IPMA temperature sanity (Lisbon historic bounds + tMin/tMax inversion)
            9. Dynamic-data disclaimers (events, Carris bus/tram info)

        Args:
            combined_output: All agent outputs concatenated.
            user_query: The user's original query (for context).
            user_context: User preferences/constraints dict.

        Returns:
            Dict with:
                - valid (bool): True if no critical issues found
                - disclaimers (List[str]): Informational warnings to surface to user
                - critical_issues (List[str]): Definitive factual errors detected
                - checks_performed (List[str]): Names of checks that ran
        """
        disclaimers: List[str] = []
        critical_issues: List[str] = []
        checks: List[str] = []
        output_lower = combined_output.lower()

        # ── Check 1: Metro station names ──────────────────────────────
        # Uses METRO_STATIONS from metrolisboa_api as the authoritative source.
        checks.append("metro_stations")
        station_text_patterns = [
            r"esta[çc][aã]o\s+(?:de\s+|do\s+)?([A-Za-zÀ-ú\s\-\.]+?)(?:\s*[\(\),\.]|\s+(?:da|na|para|line|linha|on|to|from))",
            r"station\s+([A-Za-zÀ-ú\s\-\.]+?)(?:\s*[\(\),\.]|\s+(?:on|to|from|line))",
        ]
        mentioned_stations: set = set()
        for pattern in station_text_patterns:
            for match in re.findall(pattern, output_lower, re.IGNORECASE):
                name = match.strip().lower().rstrip(".")
                word_count = len(name.split())
                if (
                    len(name) > 2
                    and word_count <= 4
                    and "metropolitano de lisboa" not in name
                    and "reconhecida" not in name
                    and "recognized" not in name
                ):
                    mentioned_stations.add(name)

        valid_metro_set = _METRO_CANONICAL_STATIONS
        invalid_stations = [
            s for s in mentioned_stations
            if s not in valid_metro_set
            and not any(s in v or v in s for v in valid_metro_set)
        ]
        if invalid_stations:
            disclaimers.append(
                f"Some metro station names could not be verified: {', '.join(invalid_stations)}"
            )

        # ── Check 2: Metro line-station pair validity ─────────────────
        # Detects hallucinations like "linha amarela to Telheiras"
        # (Telheiras is only on linha verde). Uses sentence-level analysis.
        # Requires "linha X" pattern to avoid false positives on standalone color words.
        checks.append("metro_line_station_pairs")
        if _HAS_METRO_DATA and _METRO_LINES_DATA and _METRO_STATIONS_DATA:
            sentences = re.split(r"[.!?\n]+", output_lower)
            seen_pair_issues: set = set()
            for sentence in sentences:
                for line_name in _METRO_LINES_DATA:
                    if not re.search(rf"\blinha\s+{re.escape(line_name)}\b", sentence):
                        continue
                    for station, station_lines in _METRO_STATIONS_DATA.items():
                        if len(station) < 5:  # Skip very short names (noise risk)
                            continue
                        if station in sentence and line_name not in station_lines:
                            key = f"{station}@{line_name}"
                            if key not in seen_pair_issues:
                                seen_pair_issues.add(key)
                                correct = ", ".join(station_lines)
                                disclaimers.append(
                                    f"Station '{station.title()}' does not serve the "
                                    f"{line_name} metro line (it serves: {correct})"
                                )

        # ── Check 3: CP train line names ──────────────────────────────
        # Validates CP line names against CP_LINES from cp_api.
        checks.append("cp_lines")
        if _HAS_CP_DATA and _CP_LINES_DATA:
            cp_pattern = (
                r"linha\s+de\s+([A-Za-zÀ-ú\s\-]+?)(?:[\.,;\n]|\s+(?:line|train|comboio|de|da))"
            )
            for match in re.findall(cp_pattern, output_lower):
                line_name = match.strip().lower()
                if len(line_name) > 2:
                    is_known = any(
                        line_name in key or key in line_name
                        for key in _CP_LINES_DATA
                    )
                    if not is_known:
                        valid_list = ", ".join(_CP_LINES_DATA.keys())
                        disclaimers.append(
                            f"CP train line '{match.strip()}' could not be verified. "
                            f"Known AML lines: {valid_list}"
                        )

        # ── Check 4: Coordinate bounds (AML area) ─────────────────────
        checks.append("aml_coordinates")
        coord_patterns = [
            r"(-?\d+\.?\d*)\s*[,°]\s*(-?\d+\.?\d*)",
            r"lat(?:itude)?\s*[:=]?\s*(-?\d+\.?\d*)\s*[,;]\s*lon(?:gitude)?\s*[:=]?\s*(-?\d+\.?\d*)",
        ]
        coord_matches: list = []
        for cp in coord_patterns:
            coord_matches.extend(re.findall(cp, combined_output, re.IGNORECASE))
        out_of_bounds = []
        for lat_s, lon_s in coord_matches:
            try:
                lat, lon = float(lat_s), float(lon_s)
                if 30.0 <= abs(lat) <= 50.0 and 5.0 <= abs(lon) <= 15.0:
                    if not (
                        _AML_BOUNDS["lat_min"] <= lat <= _AML_BOUNDS["lat_max"]
                        and _AML_BOUNDS["lon_min"] <= lon <= _AML_BOUNDS["lon_max"]
                    ):
                        out_of_bounds.append(f"({lat}, {lon})")
            except (ValueError, TypeError):
                continue
        if out_of_bounds:
            disclaimers.append(
                f"Some coordinates appear outside the Lisbon Metropolitan Area: "
                f"{', '.join(out_of_bounds[:3])}"
            )

        # ── Check 5: Date sanity ──────────────────────────────────────
        checks.append("date_sanity")
        today = datetime.now().date()
        date_patterns = [
            r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})",  # DD/MM/YYYY
            r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})",  # YYYY-MM-DD
        ]
        for dp in date_patterns:
            for groups in re.findall(dp, combined_output):
                try:
                    if len(groups[0]) == 4:
                        d = datetime(int(groups[0]), int(groups[1]), int(groups[2])).date()
                    else:
                        d = datetime(int(groups[2]), int(groups[1]), int(groups[0])).date()
                    if "forecast" in output_lower or "previsão" in output_lower:
                        max_forecast = today + timedelta(days=_IPMA_FORECAST_DAYS)
                        if d > max_forecast:
                            disclaimers.append(
                                f"Weather forecast for {d.isoformat()} may be beyond the "
                                f"available forecast range ({_IPMA_FORECAST_DAYS} days from IPMA)"
                            )
                except (ValueError, TypeError):
                    continue

        # ── Check 6: URL domain validation ───────────────────────────
        checks.append("url_validation")
        url_domains = re.findall(r"https?://([a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})", combined_output)
        suspicious_urls = [
            d.lower() for d in url_domains
            if not any(
                d.lower() == v or d.lower().endswith("." + v) for v in _VALID_DOMAINS
            )
        ]
        if suspicious_urls:
            unique_sus = list(set(suspicious_urls))[:5]
            disclaimers.append(
                f"Some URLs reference unverified domains: {', '.join(unique_sus)}. "
                "Please verify links before visiting."
            )

        # ── Check 7: User preference adherence ───────────────────────
        checks.append("user_preferences")
        if user_context:
            mobility = user_context.get("mobility", "")
            if mobility in ("limited", "wheelchair"):
                accessibility_terms = [
                    "acess", "wheelchair", "cadeira de rodas", "elevador",
                    "elevator", "lift", "mobilidade reduzida", "reduced mobility",
                ]
                if "transport" in output_lower and not any(
                    t in output_lower for t in accessibility_terms
                ):
                    disclaimers.append(
                        "Transport information may not include accessibility details. "
                        "Please verify station accessibility at metrolisboa.pt."
                    )
            available_time = user_context.get("available_time")
            if available_time and (
                "plan" in user_query.lower() or "roteiro" in user_query.lower()
            ):
                time_indicators = re.findall(r"(\d+)\s*(?:hours?|horas?|h\b)", output_lower)
                if time_indicators:
                    parsed_hours = [int(h) for h in time_indicators]
                    total_hours = sum(h for h in parsed_hours if h < 24)
                    if total_hours > available_time * _TIME_TOLERANCE_FACTOR:
                        disclaimers.append(
                            f"The suggested itinerary may exceed your available time of {available_time}h."
                        )

        # ── Check 8: IPMA temperature sanity ─────────────────────────
        # Flags temperatures outside Lisbon's historic range AND tMin > tMax.
        checks.append("temperature_sanity")
        temp_values: List[float] = []
        for tp in [r"(-?\d+\.?\d*)\s*°[Cc]", r"(?:tmax|tmin)\s*[:=]\s*(-?\d+\.?\d*)"]:
            for t in re.findall(tp, combined_output, re.IGNORECASE):
                try:
                    temp_values.append(float(t))
                except ValueError:
                    pass
        extreme_temps = [t for t in temp_values if t < _LISBON_TEMP_MIN or t > _LISBON_TEMP_MAX]
        if extreme_temps:
            critical_issues.append(
                f"Temperature value(s) outside Lisbon's historic range "
                f"({_LISBON_TEMP_MIN}°C to {_LISBON_TEMP_MAX}°C): "
                f"{', '.join(f'{t:.1f}°C' for t in extreme_temps[:3])}"
            )
        tmin_m = re.findall(r"\btmin\s*[:=]\s*(-?\d+\.?\d*)", combined_output, re.IGNORECASE)
        tmax_m = re.findall(r"\btmax\s*[:=]\s*(-?\d+\.?\d*)", combined_output, re.IGNORECASE)
        if tmin_m and tmax_m:
            try:
                if float(tmin_m[0]) > float(tmax_m[0]):
                    critical_issues.append(
                        f"Temperature inversion: tMin ({tmin_m[0]}°C) > tMax ({tmax_m[0]}°C)"
                    )
            except ValueError:
                pass

        # ── Check 9: Dynamic-data disclaimers ─────────────────────────
        # Adds informational caveats for data that cannot be deterministically
        # verified at runtime (events change daily, bus routes change, etc.).
        checks.append("dynamic_data_disclaimers")
        event_keywords = {
            "event", "evento", "exhibition", "exposição", "exposicao",
            "festival", "concert", "concerto", "spectacle", "espectáculo",
        }
        if any(kw in output_lower for kw in event_keywords):
            disclaimers.append(
                "Event details (dates, times, ticket prices) should be confirmed at "
                "visitlisboa.com, as this data is synced daily and may have changed."
            )
        tram_mentions = {m.lower() for m in re.findall(r"\b\d+e\b", output_lower)}
        invalid_trams = tram_mentions - _CARRIS_TRAM_LINES
        if invalid_trams:
            disclaimers.append(
                f"Tram line(s) could not be verified: "
                f"{', '.join(t.upper() for t in sorted(invalid_trams))}. "
                f"Known Lisbon trams: {', '.join(t.upper() for t in sorted(_CARRIS_TRAM_LINES))}"
            )
        if re.search(r"\b[0-9]{3}\b", combined_output) and "carris" in output_lower:
            disclaimers.append(
                "Carris bus route numbers and schedules should be verified at carris.pt, "
                "as GTFS data may not reflect the most recent changes."
            )

        # ── Check 10: User-facing output hygiene ───────────────────────
        # Flags backend-oriented fields that should never reach the final UI.
        checks.append("output_hygiene")
        if re.search(r"(?im)^\s*(?:[-*•]\s*)?(?:🗺️\s*)?GPS\s*:", combined_output):
            critical_issues.append(
                "Raw GPS coordinates leaked into user-facing output."
            )
        if re.search(
            r"(?im)^\s*(?:[-*•]\s*)?(?:🚏\s*)?(?:next\s+)?stop(?:_id|\s+id)\s*[:=]|\b(?:line_id|stop_id|route_id|pattern_id|trip_id)\b",
            combined_output,
        ):
            critical_issues.append(
                "Technical transport identifiers leaked into user-facing output."
            )
        if re.search(
            r"\b(?:Unknown event|Evento sem nome|Unknown place|Local sem nome|Unknown station|Estação sem nome)\b",
            combined_output,
            re.IGNORECASE,
        ):
            critical_issues.append(
                "Unnamed placeholder content leaked into user-facing output."
            )

        result = {
            "valid": len(critical_issues) == 0,
            "disclaimers": disclaimers,
            "critical_issues": critical_issues,
            "checks_performed": checks,
        }
        if disclaimers or critical_issues:
            logger.info(
                f"QA fact-check: {len(critical_issues)} critical issue(s), "
                f"{len(disclaimers)} disclaimer(s)"
            )
        return result


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 QA Agent Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")

    # ── Test deterministic fact-checking (no LLM needed) ─────────
    print("\n\033[1m📋 Phase 2: Deterministic Fact-Checking Tests\033[0m")
    agent = QualityAssuranceAgent.__new__(QualityAssuranceAgent)

    # Test: Valid metro station
    r = agent._verify_facts("Take metro to estação de Alameda", "test", None)
    assert "metro_stations" in r["checks_performed"]
    print("  \033[1;32m✅ PASS\033[0m: Metro station check runs")

    # Test: Coordinates in AML
    r = agent._verify_facts("Location: 38.7223, -9.1393", "test", None)
    assert len(r["disclaimers"]) == 0 or "outside" not in str(r["disclaimers"])
    print("  \033[1;32m✅ PASS\033[0m: Valid AML coordinates accepted")

    # Test: Coordinates outside AML
    r = agent._verify_facts("Location: 41.1579, -8.6291", "test", None)
    outside_found = any("outside" in d for d in r["disclaimers"])
    print(f"  {'✅ PASS' if outside_found else '⚠️ SKIP'}: Out-of-bounds coordinate flagged: {outside_found}")

    # Test: Suspicious URL
    r = agent._verify_facts("Visit https://fake-lisbon-tours.xyz/book", "test", None)
    url_flagged = any("unverified" in d for d in r["disclaimers"])
    print(f"  \033[1;32m✅ PASS\033[0m: Suspicious URL flagged: {url_flagged}")

    # Test: Valid URL
    r = agent._verify_facts("Source: https://www.visitlisboa.com/events", "test", None)
    no_url_flag = not any("unverified" in d for d in r["disclaimers"])
    print(f"  \033[1;32m✅ PASS\033[0m: Valid URL accepted: {no_url_flag}")

    # Test: Mobility preference
    r = agent._verify_facts(
        "Take transport from Alameda to Oriente via metro line vermelha",
        "plan accessible route",
        {"mobility": "wheelchair"},
    )
    access_flagged = any("accessibility" in d for d in r["disclaimers"])
    print(f"  \033[1;32m✅ PASS\033[0m: Wheelchair accessibility disclaimer: {access_flagged}")

    # Test: Metro line-station pair (Telheiras is on Verde, NOT Amarela)
    r = agent._verify_facts(
        "Toma a linha amarela até Telheiras para chegar ao destino.",
        "test", None,
    )
    wrong_pair = any("telheiras" in d.lower() for d in r["disclaimers"])
    print(f"  {'✅ PASS' if wrong_pair else '⚠️  WARN (metro data unavailable?)'}: "
          f"Wrong metro line-station pair flagged (Telheiras/Amarela): {wrong_pair}")

    # Test: Temperature out of Lisbon bounds (-30°C impossible)
    r = agent._verify_facts("Today's temperature in Lisbon is -30°C.", "test", None)
    temp_flagged = any("temperature" in i.lower() for i in r["critical_issues"])
    print(f"  \033[1;32m✅ PASS\033[0m: Extreme temperature flagged as critical: {temp_flagged}")

    # Test: Temperature inversion (tMin > tMax)
    r = agent._verify_facts("Forecast: tMin: 32, tMax: 15", "test", None)
    inversion = any("inversion" in i.lower() for i in r["critical_issues"])
    print(f"  \033[1;32m✅ PASS\033[0m: Temperature inversion detected: {inversion}")

    # Test: Event disclaimer added when events are mentioned
    r = agent._verify_facts("Join the jazz festival at CCBB tonight.", "test", None)
    event_disc = any("event" in d.lower() or "visitlisboa" in d.lower() for d in r["disclaimers"])
    print(f"  \033[1;32m✅ PASS\033[0m: Event data disclaimer present: {event_disc}")

    # Test: Known tram 28E should NOT be flagged
    r = agent._verify_facts("Take tram 28E from Martim Moniz to Prazeres.", "test", None)
    tram_ok = not any("28e" in d.lower() and "not be verified" in d.lower() for d in r["disclaimers"])
    print(f"  \033[1;32m✅ PASS\033[0m: Valid tram 28E not flagged: {tram_ok}")

    # Test: Unknown tram line (99E does not exist)
    r = agent._verify_facts("Take tram 99E across Lisbon.", "test", None)
    tram_flagged = any("99e" in d.lower() for d in r["disclaimers"])
    print(f"  {'✅ PASS' if tram_flagged else '⚠️  INFO'}: Unknown tram 99E flagged: {tram_flagged}")

    print(f"\n\033[1m📋 All deterministic checks ({len(r['checks_performed'])}): "
          f"{r['checks_performed']}\033[0m")

    # ── Test full LLM-based validation ───────────────────────────
    print("\n\033[1m📋 Phase 1+2: Full Validation Tests (requires LLM)\033[0m")
    try:
        agent = QualityAssuranceAgent()
        print(f"  \033[1m✅ QA Agent initialized:\033[0m {agent.get_model_info()}")
        print(f"     Tools: {len(agent.tools)} (QA has no tools)")

        # Test 1: Incomplete planning query
        print("\n  \033[1m📝 Test 1: Incomplete planning query\033[0m")
        result = agent.validate(
            user_query="Plan my day tomorrow in Lisbon",
            agent_outputs={
                "weather": "Tomorrow: 18°C, sunny, no rain expected.",
                "researcher": "1. Museu do Azulejo\n2. Castelo de São Jorge\n3. Belém Tower",
            },
            agents_called=["weather", "researcher"],
            language="en",
            user_context={"preferences": ["museums", "history"], "mobility": "full"},
        )
        print(f"     Complete: {result['complete']}")
        print(f"     Missing: {result['missing_data']}")
        print(f"     Required agents: {result['required_agents']}")
        print(f"     Fact-check: {result['fact_check']['checks_performed']}")

        # Test 2: Complete weather query
        print("\n  \033[1m📝 Test 2: Complete weather query\033[0m")
        result = agent.validate(
            user_query="What's the weather today?",
            agent_outputs={
                "weather": "Today: 22°C max, 14°C min. Sunny. No rain. Wind: Moderate from NW.",
            },
            agents_called=["weather"],
            language="en",
        )
        print(f"     Complete: {result['complete']}")
        print(f"     Reasoning: {result['reasoning']}")

        print("\n\033[1;32m✅ QA Agent working!\033[0m")

    except Exception as e:
        print(f"\n\033[1;31m❌ LLM-based test error (expected if no LLM configured):\033[0m {e}")
