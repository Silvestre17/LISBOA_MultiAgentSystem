# ==========================================================================
# Master Thesis - Planner Agent
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Itinerary synthesis agent. Combines outputs from other agents
#   into coherent travel plans.
# ==========================================================================

import re
import unicodedata
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from agent.agents.base import BaseAgent, clean_response, traceable
from agent.prompts.planner import get_planner_prompt
from agent.utils.response_formatter import (
    finalize_worker_response,
    infer_response_language,
)

_PLANNER_FIELD_LABELS = {
    "brief description",
    "description",
    "address",
    "location",
    "opening hours",
    "tip",
    "quick tip",
    "price",
    "prices",
    "source",
    "fonte",
    "updated",
    "atualizado",
    "conditions",
    "final notes",
    "dicas finais",
    "weather data",
    "places & attractions",
    "events",
    "transport info",
    "data limitations",
}
_PLANNER_GENERIC_ACTIVITY_TERMS = (
    "lunch",
    "dinner",
    "breakfast",
    "coffee",
    "break",
    "return",
    "free time",
    "walk",
    "transfer",
    "transport",
)
_PLANNER_PLACE_HINTS = (
    "museum",
    "museu",
    "monastery",
    "mosteiro",
    "castle",
    "castelo",
    "aqueduct",
    "lighthouse",
    "pavilion",
    "pavilh",
    "monument",
    "society",
    "science",
    "sport",
    "geographical",
    "cemetery",
    "maat",
    "mude",
    "gulbenkian",
    "berardo",
    "tower",
    "torre",
)
_PLANNER_ACCESSIBILITY_RE = re.compile(
    r"\b(wheelchair|step[- ]?free|accessible|accessibility|elevator|lift|accessible restroom|cadeira de rodas|acess[ií]vel|elevador|wc adaptado|mobilidade reduzida|curb[- ]?cut)\b",
    re.IGNORECASE,
)


def _normalize_planner_text(text: str) -> str:
    """Normalizes planner text for robust grounding comparisons."""
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = re.sub(r"[^a-zA-Z0-9\s/-]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip().lower()


def _extract_allowed_place_names(text: str) -> List[str]:
    """Extracts grounded POI names from researcher/event outputs."""
    if not text:
        return []

    candidates: List[str] = []
    seen = set()

    for line in text.splitlines():
        for match in re.findall(r"\*\*([^*]+)\*\*", line):
            candidate = match.strip().strip("-–—: ")
            normalized = _normalize_planner_text(candidate)
            if not normalized or normalized in _PLANNER_FIELD_LABELS:
                continue
            if normalized.isdigit() or re.fullmatch(r"\d+\.?", normalized):
                continue
            if len(normalized.split()) > 12:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(candidate)

    return candidates


def _query_requests_accessibility(user_message: str) -> bool:
    """Detects whether the user asked for accessibility support."""
    return bool(
        re.search(
            r"\b(wheelchair|accessible|accessibility|step[- ]?free|mobility|reduced mobility|cadeira de rodas|acess[ií]vel|mobilidade reduzida)\b",
            user_message or "",
            re.IGNORECASE,
        )
    )


def _context_has_accessibility_data(*texts: str) -> bool:
    """Returns whether accessibility details are explicitly present in context."""
    return any(_PLANNER_ACCESSIBILITY_RE.search(text or "") for text in texts)


def _clean_activity_title(title: str) -> str:
    """Removes itinerary prefixes from activity titles before validation."""
    cleaned = re.sub(
        r"^(start|optional(?:,\s*time-permitting)?|opcional(?:,\s*se houver tempo)?)\s*:\s*",
        "",
        title.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" -–—")


def _matches_allowed_place(activity_title: str, allowed_places: List[str]) -> bool:
    """Checks whether an activity title matches one of the allowed POIs."""
    normalized_activity = _normalize_planner_text(activity_title)
    if not normalized_activity:
        return True

    for place in allowed_places:
        normalized_place = _normalize_planner_text(place)
        if not normalized_place:
            continue
        if normalized_place in normalized_activity or normalized_activity in normalized_place:
            return True

    return False


def _find_planner_grounding_issues(
    response: str,
    allowed_places: List[str],
    accessibility_requested: bool,
    accessibility_confirmed: bool,
) -> List[str]:
    """Finds unsupported venue or accessibility claims in planner drafts."""
    issues: List[str] = []

    activity_lines = re.findall(r"^🕐.*?-\s*\*\*(.+?)\*\*", response or "", flags=re.MULTILINE)
    for raw_title in activity_lines:
        title = _clean_activity_title(raw_title)
        normalized_title = _normalize_planner_text(title)
        if not normalized_title:
            continue
        if any(term in normalized_title for term in _PLANNER_GENERIC_ACTIVITY_TERMS):
            continue
        if allowed_places and any(term in normalized_title for term in _PLANNER_PLACE_HINTS):
            if not _matches_allowed_place(title, allowed_places):
                issues.append(f"Unsupported venue mentioned: {title}")

    if accessibility_requested and not accessibility_confirmed and _PLANNER_ACCESSIBILITY_RE.search(response or ""):
        issues.append(
            "Accessibility details were claimed without explicit confirmation in the provided data."
        )

    return issues


def _build_planner_grounding_message(
    allowed_places: List[str],
    accessibility_requested: bool,
    accessibility_confirmed: bool,
) -> str:
    """Builds a strict grounding note for planner synthesis."""
    rules = [
        "GROUNDING RULES:",
        "- You MUST stay grounded in the provided agent data.",
        "- Do NOT mention any venue, museum, restaurant, or landmark unless it appears in the provided data.",
        "- If data is missing, say it is not confirmed instead of filling the gap from general knowledge.",
    ]

    if allowed_places:
        rules.append("- Allowed venue names: " + "; ".join(allowed_places[:15]))
        rules.append("- Any venue name not in the allowed list above is forbidden.")

    if accessibility_requested and not accessibility_confirmed:
        rules.append(
            "- Accessibility was requested, but the provided data does NOT confirm wheelchair access. "
            "Do NOT claim step-free access, elevators, accessible toilets, or wheelchair-friendly facilities. "
            "State clearly that accessibility must be confirmed with the official venue/operator."
        )

    return "\n".join(rules)


class PlannerAgent(BaseAgent):
    """
    Itinerary planner agent that synthesizes outputs from other agents.
    
    Responsibilities:
        - Combine weather, transport, and places data
        - Apply constraints (mobility, time, weather)
        - Generate coherent, practical itineraries
    
    Note:
        This agent has NO tools. It only synthesizes data gathered by worker
        agents and can surface QA disclaimers in the final planning response.
        In the default runtime, it is invoked only when the supervisor route
        includes the planner. Direct and simple single-domain queries can
        return without using this agent.
    """
    
    def __init__(self):
        """Initializes the planner agent."""
        super().__init__("planner")
        self.system_prompt = get_planner_prompt()
    
    @traceable(name="planner_agent", run_type="chain", tags=["sub-agent", "planner"])
    def invoke(
        self, 
        user_message: str, 
        weather_data: str = "",
        transport_data: str = "",
        places_data: str = "",
        events_data: str = "",
        qa_disclaimers: list[str] | None = None,
    ) -> str:
        """
        Creates an itinerary from gathered data.
        
        Args:
            user_message: The user's original query.
            weather_data: Output from weather agent.
            transport_data: Output from transport agent.
            places_data: Output from researcher agent (places).
            events_data: Output from researcher agent (events).
            qa_disclaimers: Optional list of QA-flagged data limitations.
            
        Returns:
            str: Formatted itinerary.
        """
        # Build context from agent outputs
        context_parts = []
        
        if weather_data:
            context_parts.append(f"## 🌤️ Weather Data\n{weather_data}")
        
        if places_data:
            context_parts.append(f"## 🏛️ Places & Attractions\n{places_data}")
        
        if events_data:
            context_parts.append(f"## 🎭 Events\n{events_data}")
        
        if transport_data:
            context_parts.append(f"## 🚇 Transport Info\n{transport_data}")
        
        # Inject QA disclaimers so the planner transparently communicates limitations
        if qa_disclaimers:
            disclaimer_text = "\n".join(f"- ⚠️ {d}" for d in qa_disclaimers)
            context_parts.append(
                f"## ⚠️ Data Limitations (from QA validation)\n"
                f"Include these caveats in your response where relevant:\n{disclaimer_text}"
            )
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No additional data provided."
        allowed_places = _extract_allowed_place_names("\n".join(part for part in [places_data, events_data] if part))
        accessibility_requested = _query_requests_accessibility(user_message)
        accessibility_confirmed = _context_has_accessibility_data(
            places_data,
            events_data,
            transport_data,
        )
        grounding_message = _build_planner_grounding_message(
            allowed_places=allowed_places,
            accessibility_requested=accessibility_requested,
            accessibility_confirmed=accessibility_confirmed,
        )
        
        language = infer_response_language(user_query=user_message, default="en")
        language_instruction = (
            "Respond ENTIRELY in Portuguese (PT-PT)."
            if language == "pt"
            else "Respond ENTIRELY in English."
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            SystemMessage(content=language_instruction),
            SystemMessage(content=grounding_message),
            SystemMessage(content=f"# Data from Specialized Agents\n\n{context}"),
            HumanMessage(content=f"Based on the data above, create an itinerary for: {user_message}")
        ]
        
        # Planner has no tools - LLM call with retry for Azure content filter
        response = self._safe_llm_invoke(self.llm, messages)
        cleaned_response = clean_response(response.content)

        grounding_issues = _find_planner_grounding_issues(
            cleaned_response,
            allowed_places=allowed_places,
            accessibility_requested=accessibility_requested,
            accessibility_confirmed=accessibility_confirmed,
        )

        retry_count = 0
        while grounding_issues and retry_count < 2:
            retry_count += 1
            retry_messages = messages + [
                SystemMessage(
                    content=(
                        "Your previous draft violated the grounding rules. Revise it now.\n"
                        "- Remove any unsupported venue names.\n"
                        "- Remove unsupported accessibility claims.\n"
                        "- Keep only facts grounded in the provided data."
                    )
                ),
                HumanMessage(
                    content=(
                        "Revise this itinerary draft and fix the grounding issues below.\n\n"
                        "Grounding issues:\n- " + "\n- ".join(grounding_issues) +
                        "\n\nDraft:\n" + cleaned_response
                    )
                ),
            ]
            response = self._safe_llm_invoke(self.llm, retry_messages)
            cleaned_response = clean_response(response.content)
            grounding_issues = _find_planner_grounding_issues(
                cleaned_response,
                allowed_places=allowed_places,
                accessibility_requested=accessibility_requested,
                accessibility_confirmed=accessibility_confirmed,
            )

        return finalize_worker_response(
            cleaned_response,
            agent_name="planner",
            user_query=user_message,
            language=language,
        )
    
    def synthesize(self, user_message: str, agent_outputs: Dict[str, str]) -> str:
        """
        Synthesizes outputs from multiple agents into a response.
        
        Extracts QA disclaimers from internal keys and passes them
        to the planner so data limitations are surfaced to the user.
        
        Args:
            user_message: Original user query.
            agent_outputs: Dict mapping agent names to their outputs.
                May contain '_qa_disclaimers' (list) from QA validation.
            
        Returns:
            str: Synthesized response.
        """
        # Extract QA disclaimers before passing to invoke
        qa_disclaimers = agent_outputs.get("_qa_disclaimers")
        if isinstance(qa_disclaimers, str):
            # Safety: if it was stored as a string, wrap in list
            qa_disclaimers = [qa_disclaimers]

        return self.invoke(
            user_message=user_message,
            weather_data=agent_outputs.get("weather", ""),
            transport_data=agent_outputs.get("transport", ""),
            places_data=agent_outputs.get("researcher", ""),
            events_data="",  # Events come from researcher too
            qa_disclaimers=qa_disclaimers,
        )


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Planner Agent Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    try:
        agent = PlannerAgent()
        print(f"\n\033[1m✅ Planner Agent initialized:\033[0m {agent.get_model_info()}")
        print(f"   Tools: {len(agent.tools)} (planner has no tools)")
        
        # Simulate data from other agents
        mock_weather = """
        Today in Lisbon: ☀️ Clear sky
        🌡️ Temperature: 18°C - 24°C
        🌧️ Precipitation: 10% (unlikely)
        🌤️ UV Index: High - bring sunscreen!
        """
        
        mock_places = """
        1. 🏛️ **Mosteiro dos Jerónimos** - UNESCO World Heritage
           📍 Belém | 🕐 10:00-17:00 | 💰 €10
        
        2. 🏛️ **Museu Nacional dos Coches** - World's best carriage collection
           📍 Belém | 🕐 10:00-18:00 | 💰 €8
        
        3. 🎨 **MAAT** - Modern architecture & contemporary art
           📍 Belém | 🕐 11:00-19:00 | 💰 €9
        """
        
        print("\n\033[1m📝 Testing with mock data:\033[0m")
        response = agent.invoke(
            user_message="Plan my morning in Belém",
            weather_data=mock_weather,
            places_data=mock_places
        )
        print("\n\033[1m🤖 Response:\033[0m")
        print(response[:800] + "..." if len(response) > 800 else response)
        
        print("\n\033[1;32m✅ Planner agent working!\033[0m")
        
    except Exception as e:
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
        import traceback
        traceback.print_exc()
