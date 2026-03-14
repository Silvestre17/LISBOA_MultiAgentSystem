# ==========================================================================
# Master Thesis - Researcher Agent
#   - André Filipe Gomes Silvestre, 20240502
#
#   RAG-based researcher for places, events, and local knowledge.
#   Uses semantic search over vector store.
#   Uses BaseAgent.execute_react_loop() for tool execution.
# ==========================================================================

import re
import uuid
from typing import TYPE_CHECKING, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from agent.agents.base import BaseAgent, traceable
from agent.prompts.researcher import get_researcher_prompt
from agent.state import AgentState
from agent.utils.langgraph_compat import ToolNode
from agent.utils.response_formatter import (
    finalize_worker_response,
    infer_response_language,
)


class ResearcherAgent(BaseAgent):
    """
    RAG researcher agent for places, events, and local knowledge.

    Uses 11 retrieval tools loaded via get_agent_tools:
        - search_places_attractions
        - search_cultural_events
        - search_lisbon_knowledge
        - find_nearby_services (pharmacies, hospitals, etc.)
        - get_event_categories / get_place_categories
        - search_history_culture (web search for history/facts)
        - list_available_datasets / get_dataset_details / find_place_in_datasets
        - list_service_categories

    Notes:
        This agent combines semantic retrieval over the vector store, on-demand
        open-data lookup, and web fallback search. It is the main worker for
        places, events, essential services, and Lisbon knowledge queries.
    """

    def __init__(self):
        """Initializes the researcher agent."""
        super().__init__("researcher")
        self.system_prompt = get_researcher_prompt()
        # Tools are loaded by BaseAgent.__init__ via get_agent_tools("researcher")
        # which returns the full set including dados_abertos tools

    @staticmethod
    def _is_content_filter_error(error: Exception) -> bool:
        """Returns whether an exception is an Azure content-filter false positive."""
        error_str = str(error).lower()
        return (
            "content_filter" in error_str
            or "responsibleaipolicyviolation" in error_str
            or "jailbreak" in error_str
        )

    @staticmethod
    def _infer_research_query_language(user_message: str) -> str:
        """Adds a light PT-PT heuristic for broad local-discovery prompts."""
        query = (user_message or "").lower()
        query_tokens = set(re.findall(r"[a-zà-ÿ]+", query))
        pt_token_markers = {
            "quero", "lista", "atrações", "atracoes", "imperdíveis", "imperdiveis",
            "hoje", "amanhã", "amanha", "eventos", "locais", "monumentos",
            "miradouro", "museu", "museus",
        }
        en_token_markers = {
            "what", "where", "best", "museum", "museums", "events", "attractions",
            "wheelchair", "accessible", "history", "places", "nearby",
        }
        if any(token in query_tokens for token in pt_token_markers) or "esta semana" in query or re.search(r"[ãõáéíóúç]", query):
            return "pt"
        if any(token in query_tokens for token in en_token_markers) or any(
            phrase in query for phrase in ["first time", "must see", "must-see", "in lisbon", "in belem", "in belém", "near "]
        ):
            return "en"
        return infer_response_language(user_query=user_message, default="en")

    @staticmethod
    def _build_messages(system_prompt: str, user_message: str, context: str = "") -> list:
        """Builds the message list for a researcher invocation."""
        language = ResearcherAgent._infer_research_query_language(user_message)
        language_instruction = (
            "Respond ENTIRELY in Portuguese (PT-PT)."
            if language == "pt"
            else "Respond ENTIRELY in English."
        )

        messages = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=language_instruction),
        ]

        if context:
            messages.append(SystemMessage(content=f"Context from other agents:\n{context}"))

        messages.append(HumanMessage(content=user_message))
        return messages

    def _get_tool_by_name(self, tool_name: str):
        """Returns a loaded tool by name, or None if not found."""
        for tool in self.tools:
            if getattr(tool, "name", "") == tool_name:
                return tool
        return None

    @staticmethod
    def _is_accessibility_place_query(user_message: str) -> bool:
        """Detects high-risk accessibility place queries that should skip free-form synthesis."""
        query = (user_message or "").lower()
        accessibility_terms = [
            "wheelchair", "accessible", "accessibility", "step-free",
            "cadeira de rodas", "acessível", "acessivel", "mobilidade reduzida",
        ]
        place_terms = [
            "museum", "museu", "monument", "monumento", "place", "places",
            "attraction", "attractions", "belem", "belém",
        ]
        return any(term in query for term in accessibility_terms) and any(term in query for term in place_terms)

    def _run_accessibility_place_lookup(self, user_message: str, language: str) -> str:
        """Runs a deterministic place lookup for accessibility-focused queries."""
        tool = self._get_tool_by_name("search_places_attractions")
        if not tool:
            return self._run_direct_tool_fallback(user_message, language)

        args = {"query": user_message, "max_results": 5}
        if any(term in user_message.lower() for term in ["museum", "museu", "monument", "monumento"]):
            args["category"] = "Museums & Monuments"

        result = tool.invoke(args)
        source_line = self._build_places_source_line(result, language)
        return f"{result}\n\n{source_line}".strip()

    @staticmethod
    def _build_places_source_line(result: str, language: str) -> str:
        """Builds the right source line for direct place lookups, including hybrid open-data results."""
        if "Open Data:" in result or re.search(r"\b[1-9]\d*\s+from Lisboa Aberta\b", result or ""):
            if language == "pt":
                return "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais) e [*Lisboa Aberta*](https://dados.cm-lisboa.pt/)"
            return "📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places) and [*Lisboa Aberta*](https://dados.cm-lisboa.pt/)"

        if language == "pt":
            return "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais)"
        return "📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places)"

    @staticmethod
    def _build_events_source_line(language: str) -> str:
        """Builds the right source line for direct event lookups."""
        if language == "pt":
            return "📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)"
        return "📌 **Source:** [*VisitLisboa Events*](https://www.visitlisboa.com/en/events)"

    @staticmethod
    def _is_broad_attractions_query(user_message: str) -> bool:
        """Detects broad attraction-list queries that should bypass free-form synthesis."""
        query = (user_message or "").lower()
        attraction_phrases = [
            "atrações imperdíveis",
            "atracoes imperdiveis",
            "atrações",
            "atracoes",
            "must-see",
            "must see",
            "first time",
            "primeira vez",
            "top attractions",
            "highly recommended attractions",
            "main attractions",
            "o que visitar",
            "what should i visit",
        ]
        planning_terms = ["itinerary", "roteiro", "plan", "plano", "schedule", "agenda"]
        return any(phrase in query for phrase in attraction_phrases) and not any(
            term in query for term in planning_terms
        )

    @staticmethod
    def _is_direct_event_lookup_query(user_message: str) -> bool:
        """Detects event-discovery queries that are safer to answer directly from tools."""
        query = (user_message or "").lower()
        planning_terms = [
            "plan", "plano", "roteiro", "itinerary", "agenda",
            "combine", "combinar", "day plan", "plan my day",
        ]
        event_terms = [
            "event", "events", "evento", "eventos", "concert", "concerto",
            "festival", "festivals", "exhibition", "exposição", "exposicao",
            "music", "música", "musica", "show", "theatre", "teatro",
            "dance", "dança", "danca", "cinema", "what's on", "o que há", "o que ha",
        ]
        return any(term in query for term in event_terms) and not any(
            term in query for term in planning_terms
        )

    @staticmethod
    def _extract_event_date_filter(user_message: str) -> Optional[str]:
        """Extracts a lightweight date filter for direct event tool lookups."""
        query = (user_message or "").lower()
        mappings = [
            (["this weekend", "este fim de semana", "fim de semana"], "this weekend"),
            (["next week", "próxima semana", "proxima semana"], "next week"),
            (["this week", "esta semana"], "this week"),
            (["tomorrow", "amanhã", "amanha"], "tomorrow"),
            (["today", "hoje"], "today"),
            (["next month", "próximo mês", "proximo mes"], "next month"),
            (["this month", "este mês", "este mes"], "this month"),
        ]
        for terms, date_filter in mappings:
            if any(term in query for term in terms):
                return date_filter
        return None

    @staticmethod
    def _extract_event_focus_query(user_message: str) -> Optional[str]:
        """Drops generic event phrasing so broad date-based event searches keep high recall."""
        query = (user_message or "").lower()
        specific_interest_terms = [
            "music", "música", "musica", "concert", "concerto", "concertos", "fado",
            "jazz", "rock", "pop", "festival", "festivais", "exhibition", "exposição",
            "exposicao", "theatre", "teatro", "dance", "dança", "danca", "cinema",
            "art", "arte", "family", "família", "familia", "kids", "children",
            "child", "miúdos", "miudos", "crianças", "criancas", "night",
            "nightlife", "evening", "noite", "food", "gastronomia",
            "sports", "desporto", "desportos", "market", "mercado", "fair", "feira",
        ]
        if not any(term in query for term in specific_interest_terms):
            return None

        generic_terms = {
            "que", "quais", "what", "which", "major", "great", "grandes", "large",
            "event", "events", "evento", "eventos", "this", "week", "esta", "semana",
            "este",
            "today", "hoje", "tomorrow", "amanhã", "amanha", "next", "weekend",
            "fim", "de", "semana", "local", "locais", "culture", "cultura", "cultural",
            "explore", "explorar", "lisbon", "lisboa", "temos", "there", "happening", "have",
            "algo", "interessante", "fazer", "para", "perto", "near", "around", "theres",
        }
        tokens = [token for token in re.findall(r"[a-zA-ZÀ-ÿ0-9]+", query) if len(token) >= 3]
        meaningful_tokens = [token for token in tokens if token not in generic_terms]
        return " ".join(dict.fromkeys(meaningful_tokens)) if meaningful_tokens else None

    @staticmethod
    def _is_direct_place_lookup_query(user_message: str) -> bool:
        """Detects straightforward place and service lookups that are safer to answer directly from tools."""
        query = (user_message or "").lower()
        history_keywords = ["history", "história", "historia", "culture", "cultura"]
        event_keywords = [
            "event", "events", "evento", "eventos", "concert", "concerto",
            "festival", "exhibition", "exposição", "exposicao", "show",
        ]
        place_keywords = [
            "museum", "museu", "monument", "monumento", "restaurant", "restaurante",
            "hotel", "viewpoint", "miradouro", "beach", "praia", "garden", "jardim",
            "park", "parque", "hospital", "pharmacy", "farmácia", "farmacia", "school",
            "escola", "library", "biblioteca", "police", "polícia", "policia",
            "attraction", "attractions", "place", "places", "where is", "onde fica",
            "belém", "belem",
        ]
        service_keywords = [
            "hospital", "pharmacy", "farmácia", "farmacia", "school", "escola",
            "library", "biblioteca", "police", "polícia", "policia",
        ]
        specific_lookup_markers = [
            "where is", "onde fica", "near ", "perto ", "belém", "belem", "alfama",
            "chiado", "baixa", "rossio", "oriente", "ajuda", "bairro alto", "cais do sodré",
            "cais do sodre", "campo grande", "colombo", "gulbenkian",
        ]
        if any(keyword in query for keyword in history_keywords + event_keywords):
            return False
        if any(keyword in query for keyword in service_keywords):
            return True
        return any(keyword in query for keyword in place_keywords) and any(marker in query for marker in specific_lookup_markers)

    @staticmethod
    def _extract_near_location_name(user_message: str) -> Optional[str]:
        """Extracts a nearby-location target from simple PT/EN service phrasings."""
        patterns = [
            r"\bnear\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bnear the\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bperto de\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bperto do\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
            r"\bperto da\s+(?P<location>.+?)(?:[\?\!\.,]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, user_message, flags=re.IGNORECASE)
            if match:
                return match.group("location").strip(" .?!,")
        return None

    def _run_direct_place_lookup(self, user_message: str, language: str) -> str:
        """Runs a deterministic tool path for simple place and service lookups."""
        message_lower = user_message.lower()
        places_tool = self._get_tool_by_name("search_places_attractions")
        nearby_tool = self._get_tool_by_name("find_nearby_services")

        service_keywords = [
            "pharmacy", "farmácia", "farmacia", "hospital", "school", "escola",
            "library", "biblioteca", "park", "garden", "jardim", "police", "polícia", "policia",
        ]
        if nearby_tool and any(keyword in message_lower for keyword in service_keywords):
            nearby_location = self._extract_near_location_name(user_message)
            if nearby_location:
                return nearby_tool.invoke(
                    {
                        "service_type": self._extract_service_type(user_message),
                        "near_location_name": nearby_location,
                        "max_results": 5,
                    }
                )

        if not places_tool:
            return self._run_direct_tool_fallback(user_message, language)

        query_text = user_message
        max_results = 5
        if self._is_broad_attractions_query(user_message):
            query_text = f"{user_message} iconic monuments museums palaces castles historic sites"
            max_results = 6

        args = {"query": query_text, "max_results": max_results}
        if self._is_broad_attractions_query(user_message):
            args["category"] = "Museums & Monuments"
        elif any(term in message_lower for term in ["museum", "museu", "monument", "monumento"]):
            args["category"] = "Museums & Monuments"

        result = places_tool.invoke(args)
        source_line = self._build_places_source_line(result, language)
        return f"{result}\n\n{source_line}".strip()

    def _run_direct_event_lookup(self, user_message: str, language: str) -> str:
        """Runs a deterministic VisitLisboa event lookup with explicit date parsing."""
        events_tool = self._get_tool_by_name("search_cultural_events")
        if not events_tool:
            return (
                "Não consegui aceder à pesquisa de eventos neste momento."
                if language == "pt"
                else "I couldn't access the event search tool right now."
            )

        args = {"max_results": 5, "language": language}
        date_filter = self._extract_event_date_filter(user_message)
        focus_query = self._extract_event_focus_query(user_message)

        if date_filter:
            args["date_filter"] = date_filter
        if focus_query:
            args["query"] = focus_query

        result = events_tool.invoke(args)
        source_line = self._build_events_source_line(language)
        return f"{result}\n\n{source_line}".strip()

    @staticmethod
    def _extract_service_type(user_message: str) -> str:
        """Extracts a practical service keyword for direct open-data fallback."""
        query = user_message.lower()
        service_map = {
            "pharmacy": "farmácias",
            "farmácia": "farmácias",
            "farmacias": "farmácias",
            "hospital": "hospitais",
            "hospitais": "hospitais",
            "school": "escolas",
            "escola": "escolas",
            "library": "bibliotecas",
            "biblioteca": "bibliotecas",
            "park": "jardins",
            "jardim": "jardins",
            "garden": "jardins",
            "police": "polícia",
            "polícia": "polícia",
        }

        for keyword, service_type in service_map.items():
            if keyword in query:
                return service_type

        return user_message

    def _run_direct_tool_fallback(self, user_message: str, language: str) -> str:
        """
        Runs a deterministic tool-only fallback when Azure blocks both prompt
        attempts. This avoids failing benign queries like 'Museums in Lisbon'.
        """
        message_lower = user_message.lower()

        service_keywords = [
            "pharmacy", "farmácia", "farmacias", "hospital", "school",
            "escola", "library", "biblioteca", "park", "garden",
            "jardim", "police", "polícia",
        ]
        history_keywords = ["history", "história", "historia", "culture", "cultura"]
        event_keywords = [
            "event", "events", "evento", "eventos", "concert", "concerto",
            "festival", "exhibition", "exposição", "exposicao", "show",
        ]
        category_keywords = ["categories", "categorias", "service categories", "tipos de serviços"]

        if any(keyword in message_lower for keyword in category_keywords):
            tool = self._get_tool_by_name("list_service_categories")
            if tool:
                return tool.invoke({})

        if any(keyword in message_lower for keyword in service_keywords):
            tool = self._get_tool_by_name("find_nearby_services")
            if tool:
                return tool.invoke({
                    "service_type": self._extract_service_type(user_message),
                    "max_results": 5,
                })

        if any(keyword in message_lower for keyword in history_keywords):
            tool = self._get_tool_by_name("search_history_culture")
            if tool:
                return tool.invoke({"query": user_message, "language": language})

        if any(keyword in message_lower for keyword in event_keywords):
            return self._run_direct_event_lookup(user_message, language)

        tool = self._get_tool_by_name("search_places_attractions")
        if tool:
            result = tool.invoke({"query": user_message, "max_results": 5})
            source_line = self._build_places_source_line(result, language)
            return f"{result}\n\n{source_line}".strip()

        fallback_text = (
            "I couldn't complete the semantic search prompt flow, but the retrieval tools are available."
            if language == "en"
            else "Não consegui concluir o fluxo semântico do prompt, mas as ferramentas de pesquisa continuam disponíveis."
        )
        return fallback_text

    @staticmethod
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

    @staticmethod
    def _build_language_instruction(language: str) -> str:
        """Builds a compact language instruction for subgraph LLM steps."""
        return (
            "Respond ENTIRELY in Portuguese (PT-PT)."
            if language == "pt"
            else "Respond ENTIRELY in English."
        )

    def _ensure_subgraph_messages(self, messages: list, language: str) -> list:
        """Ensures researcher subgraph LLM calls receive system and language instructions."""
        updated_messages = list(messages)
        if not updated_messages or not isinstance(updated_messages[0], SystemMessage):
            updated_messages = [SystemMessage(content=self.system_prompt)] + updated_messages

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

    @classmethod
    def _build_deterministic_subgraph_tool_call(cls, user_message: str) -> Optional[AIMessage]:
        """Routes obvious researcher queries to their canonical tool in the subgraph."""
        query = user_message.strip()
        query_lower = query.lower()

        if "search the web for the history of" in query_lower:
            subject = re.sub(r"^.*history of\s+", "", query, flags=re.IGNORECASE).strip(" .?!")
            return cls._build_tool_call("search_history_culture", {"query": subject or query, "language": "en"})

        if "service categories" in query_lower:
            return cls._build_tool_call("list_service_categories", {})

        if "dataset details for" in query_lower:
            dataset_name = re.sub(r"^.*dataset details for\s+", "", query, flags=re.IGNORECASE).strip(" .?!")
            return cls._build_tool_call("get_dataset_details", {"dataset_name": dataset_name or query})

        if "open datasets for" in query_lower:
            place_query = re.sub(r"^.*open datasets for\s+", "", query, flags=re.IGNORECASE).strip(" .?!")
            return cls._build_tool_call("find_place_in_datasets", {"query": place_query or query})

        if "list available lisboa aberta service datasets" in query_lower:
            return cls._build_tool_call("list_available_datasets", {})

        if "event categories" in query_lower:
            return cls._build_tool_call("get_event_categories", {})

        if "place categories" in query_lower:
            return cls._build_tool_call("get_place_categories", {})

        if "knowledge base for" in query_lower:
            search_query = re.sub(r"^.*knowledge base for\s+", "", query, flags=re.IGNORECASE).strip(" .?!")
            return cls._build_tool_call("search_lisbon_knowledge", {"query": search_query or query, "max_results": 5})

        if query_lower.startswith("encontra farmácias perto do"):
            place_name = re.sub(r"^encontra farmácias perto do\s+", "", query, flags=re.IGNORECASE).strip(" .?!")
            return cls._build_tool_call("find_nearby_services", {"service_type": "farmácias", "near_location_name": place_name or "Rossio", "max_results": 5})

        if "cultural events" in query_lower or "events in lisbon" in query_lower:
            date_filter = "this weekend" if "this weekend" in query_lower else "today" if "today" in query_lower else None
            args = {"query": "cultural events", "max_results": 5}
            if date_filter:
                args["date_filter"] = date_filter
            return cls._build_tool_call("search_cultural_events", args)

        if "attractions related to" in query_lower:
            search_query = re.sub(r"^.*attractions related to\s+", "", query, flags=re.IGNORECASE).strip(" .?!")
            return cls._build_tool_call("search_places_attractions", {"query": search_query or query, "max_results": 5})

        place_keywords = [
            "museum", "museu", "monument", "monumento", "restaurant", "restaurante",
            "hotel", "viewpoint", "miradouro", "beach", "praia", "garden", "jardim",
            "park", "parque", "hospital", "pharmacy", "farmácia", "farmacia", "school",
            "escola", "library", "biblioteca", "police", "polícia", "policia", "belém", "belem",
        ]
        if any(keyword in query_lower for keyword in place_keywords):
            args = {"query": query, "max_results": 5}
            if any(term in query_lower for term in ["museum", "museu", "monument", "monumento"]):
                args["category"] = "Museums & Monuments"
            return cls._build_tool_call("search_places_attractions", args)

        return None

    @traceable(name="researcher_agent", run_type="chain", tags=["sub-agent", "researcher"])
    def invoke(
        self, user_message: str, context: str = "", verbose: bool = False
    ) -> str:
        """
        Processes a places/events query using semantic search.

        Args:
            user_message: The user's query.
            context: Additional context from other agents (optional).
            verbose: Whether involved tool calls should be printed.

        Returns:
            str: Places/events information response.
        """
        # Extract explicit language preference from context if provided
        import re
        language_match = re.search(r"User language:\s*(en|pt)", context, re.IGNORECASE)
        if language_match:
            language = language_match.group(1).lower()
        else:
            language = self._infer_research_query_language(user_message)
            
        messages = self._build_messages(self.system_prompt, user_message, context)

        # Skip tool enforcement for greetings/thanks
        is_greeting = any(
            w in user_message.lower()
            for w in ["hello", "thanks", "obrigado", "tchau", "olá", "bom dia"]
        )

        tool_enforcement_msg = "" if is_greeting else (
            "You MUST use a tool (like search_places_attractions) to get real data. "
            "Do NOT answer from your knowledge base. Call the tool now."
        )

        if not is_greeting and self._is_accessibility_place_query(user_message):
            if verbose:
                print("      [RESEARCHER] Using deterministic place lookup for accessibility-focused query...")

            response = self._run_accessibility_place_lookup(user_message, language)
            return finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            )

        if not is_greeting and self._is_direct_event_lookup_query(user_message):
            if verbose:
                print("      [RESEARCHER] Using deterministic direct event lookup...")

            response = self._run_direct_event_lookup(user_message, language)
            return finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            )

        if not is_greeting and hasattr(self, "tools") and (
            self._is_direct_place_lookup_query(user_message)
            or self._is_broad_attractions_query(user_message)
        ):
            if verbose:
                print("      [RESEARCHER] Using deterministic direct place lookup...")

            response = self._run_direct_place_lookup(user_message, language)
            return finalize_worker_response(
                response,
                agent_name="researcher",
                user_query=user_message,
                language=language,
            )

        try:
            response = self.execute_react_loop(
                messages=messages,
                verbose=verbose,
                max_iterations=5,
                tool_enforcement_msg=tool_enforcement_msg,
            )
        except Exception as e:
            if not self._is_content_filter_error(e):
                raise

            if verbose:
                print("      [RESEARCHER] Retrying with safe prompt variant after content filter...")

            safe_messages = self._build_messages(
                get_researcher_prompt(safe_mode=True),
                user_message,
                context,
            )
            try:
                response = self.execute_react_loop(
                    messages=safe_messages,
                    verbose=verbose,
                    max_iterations=5,
                    tool_enforcement_msg=tool_enforcement_msg,
                )
            except Exception as safe_error:
                if not self._is_content_filter_error(safe_error):
                    raise

                if verbose:
                    print("      [RESEARCHER] Falling back to direct tool invocation after repeated content-filter blocks...")

                response = self._run_direct_tool_fallback(user_message, language)

        return finalize_worker_response(
            response,
            agent_name="researcher",
            user_query=user_message,
            language=language,
        )

    def build_subgraph(self) -> "CompiledStateGraph":
        """
        Builds a LangGraph subgraph for this agent.

        Returns:
            CompiledStateGraph: Compiled subgraph for researcher queries.
        """

        def agent_node(state: AgentState) -> dict:
            """Researcher agent decision node."""
            messages = list(state["messages"])

            user_message = None
            for message in reversed(messages):
                if isinstance(message, HumanMessage) and message.content:
                    user_message = str(message.content)
                    break

            language = infer_response_language(user_query=user_message or "", default="en")

            last_message = messages[-1] if messages else None
            if isinstance(last_message, ToolMessage):
                response = self._safe_llm_invoke(
                    self.llm_with_tools,
                    self._ensure_subgraph_messages(messages, language),
                )
                return {"messages": [response]}

            if user_message:
                deterministic_call = self._build_deterministic_subgraph_tool_call(user_message)
                if deterministic_call is not None:
                    return {"messages": [deterministic_call]}

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
        workflow.add_node("tools", ToolNode(self.tools))
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
    print("\033[1m🧪 Researcher Agent Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")

    try:
        agent = ResearcherAgent()
        print(
            f"\n\033[1m✅ Researcher Agent initialized:\033[0m {agent.get_model_info()}"
        )
        print(f"   Tools: {[t.name for t in agent.tools]}")

        print("\n\033[1m📝 Testing query:\033[0m 'Museums in Lisbon'")
        response = agent.invoke("Museums in Lisbon")
        print("\n\033[1m🤖 Response:\033[0m")
        print(response)

        print("\n\033[1;32m✅ Researcher agent working!\033[0m")

    except Exception as e:
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
        import traceback

        traceback.print_exc()
