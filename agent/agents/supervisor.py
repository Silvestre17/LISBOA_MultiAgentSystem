# ==========================================================================
# Master Thesis - Supervisor Agent
#   - André Filipe Gomes Silvestre, 20240502
#
#   Smart router that analyzes user intent and decides which specialized
#   agents to invoke. Only calls agents when necessary.
# ==========================================================================

import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.agents.base import BaseAgent, clean_response, parse_json_response, traceable
from agent.prompts.supervisor import get_supervisor_prompt
from agent.utils.response_formatter import strip_unsupported_closing_offers


class SupervisorAgent(BaseAgent):
    """
    Supervisor agent that routes queries to specialized agents.

    Responsibilities:
        - Analyze user query intent
        - Decide which agents to call (can be 0, 1, or multiple)
        - Handle simple queries directly without calling agents
        - Return routing decisions as structured JSON
    """

    def __init__(self):
        """Initializes the supervisor agent."""
        super().__init__("supervisor")
        # System prompt is now dynamic per request

    @staticmethod
    def _normalize_query(user_message: str) -> str:
        """Normalizes user text for lightweight routing heuristics."""
        return re.sub(r"[!?.,;:]+", "", (user_message or "").strip().lower())

    @classmethod
    def _is_greeting_only(cls, user_message: str) -> bool:
        """Detects greeting-only messages that should bypass the LLM."""
        normalized = cls._normalize_query(user_message)
        return normalized in {
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "olá",
            "ola",
            "bom dia",
            "boa tarde",
            "boa noite",
            "thanks",
            "thank you",
            "obrigado",
            "obrigada",
        }

    @staticmethod
    def _has_lisbon_context(message_lower: str) -> bool:
        """Returns whether the query clearly references Lisbon/AML topics."""
        lisbon_keywords = [
            "lisbon",
            "lisboa",
            "aml",
            "metro",
            "bus",
            "autocarro",
            "comboio",
            "transport",
            "transporte",
            "weather",
            "tempo",
            "forecast",
            "previsão",
            "museum",
            "museu",
            "event",
            "evento",
            "restaurant",
            "restaurante",
            "pharmacy",
            "farmácia",
            "hospital",
            "route",
            "rota",
            "belém",
            "belem",
            "chiado",
            "alfama",
            "bairro alto",
            "rossio",
            "oriente",
            "sintra",
            "cascais",
        ]
        return any(keyword in message_lower for keyword in lisbon_keywords)

    @staticmethod
    def _looks_like_weather_query(message_lower: str) -> bool:
        """Detects weather queries without over-matching generic PT words like `tempo`."""
        weather_patterns = [
            r"\bweather\b",
            r"\brain\b",
            r"\btemperature\b",
            r"\bforecast\b",
            r"\bmeteo\b",
            r"\bchover\b",
            r"\bprevis[aã]o\b",
            r"\bchuva\b",
            r"\btemperatura\b",
            r"\bsol\b",
            r"\bcomo est[aá] o tempo\b",
            r"\bqual (?:é|e) o tempo\b",
            r"\btempo em\b",
        ]
        return any(re.search(pattern, message_lower) for pattern in weather_patterns)

    @classmethod
    def _is_obvious_out_of_scope(cls, user_message: str) -> bool:
        """Detects clearly out-of-scope trivia, math, coding, or translation queries."""
        message_lower = (user_message or "").lower()
        if cls._has_lisbon_context(message_lower):
            return False

        out_of_scope_patterns = [
            r"\b\d+\s*[-+*/x]\s*\d+\b",
            r"\bcapital of\b",
            r"\bpresident of\b",
            r"\bwho won\b",
            r"\bhow do you say\b",
            r"\btranslate\b",
            r"\bcomo se diz\b",
            r"\btradu[zç][aã]o\b",
            r"\bwrite code\b",
            r"\bcoding\b",
            r"\bprogramming\b",
            r"\bpython\b",
            r"\bjavascript\b",
            r"\bsql\b",
            r"\bmandarim\b",
            r"\bjapan\b",
            r"\bjap[aã]o\b",
        ]
        return any(re.search(pattern, message_lower) for pattern in out_of_scope_patterns)

    @staticmethod
    def _build_greeting_response(language: str) -> str:
        """Builds a lightweight greeting response."""
        if language == "en":
            return (
                "Hello! 👋 I'm your Lisbon Urban Assistant. "
                "How can I help you today? I can help with weather, transport, places, events, and itineraries around Lisbon."
            )
        return (
            "Olá! 👋 Sou o teu Assistente Urbano de Lisboa. "
            "Em que te posso ajudar hoje? Posso ajudar com meteorologia, transportes, locais, eventos e itinerários em Lisboa."
        )

    @staticmethod
    def _build_out_of_scope_response(language: str) -> str:
        """Builds a friendly out-of-scope redirection response."""
        if language == "en":
            return (
                "Oops, that's outside my area of expertise! 😄 "
                "I'm your **Lisbon Urban Assistant** and I focus on the Lisbon Metropolitan Area 🏙️\n\n"
                "Here's what I can help you with:\n\n"
                "- 🌤️ Weather forecasts and real-time warnings\n"
                "- 🚇 Transport information (Metro, buses, trains, trams)\n"
                "- 🎭 Cultural events and activities\n"
                "- 📍 Places to visit, restaurants, and attractions\n"
                "- 🗺️ Custom itinerary planning\n"
                "- 🏥 Nearby services such as pharmacies and hospitals\n"
                "- 📚 Lisbon history and culture\n\n"
                "Ask me anything about Lisbon and I'll jump in. 🧭"
            )
        return (
            "Ups, isso fica fora da minha especialidade! 😄 "
            "Sou o teu **Assistente Urbano de Lisboa** e foco-me na Área Metropolitana de Lisboa 🏙️\n\n"
            "Posso ajudar-te com:\n\n"
            "- 🌤️ Previsões meteorológicas e avisos em tempo real\n"
            "- 🚇 Informação de transportes (Metro, autocarros, comboios, elétricos)\n"
            "- 🎭 Eventos culturais e atividades\n"
            "- 📍 Locais para visitar, restaurantes e atrações\n"
            "- 🗺️ Planeamento de itinerários à medida\n"
            "- 🏥 Serviços próximos, como farmácias e hospitais\n"
            "- 📚 História e cultura de Lisboa\n\n"
            "Pergunta-me o que quiseres sobre Lisboa e eu trato disso. 🧭"
        )

    def _direct_routing_override(self, user_message: str, language: str) -> Optional[Dict[str, Any]]:
        """Handles trivial direct responses before invoking the supervisor LLM."""
        if self._is_greeting_only(user_message):
            return {
                "reasoning": "Direct greeting override",
                "agents": [],
                "direct_response": self._sanitize_direct_response(self._build_greeting_response(language)),
            }

        if self._is_obvious_out_of_scope(user_message):
            return {
                "reasoning": "Direct out-of-scope override",
                "agents": [],
                "direct_response": self._sanitize_direct_response(self._build_out_of_scope_response(language)),
            }

        return None

    @staticmethod
    def _sanitize_direct_response(text: Optional[str]) -> Optional[str]:
        """Removes unsupported closing offers from direct supervisor responses."""
        if not text:
            return text
        return strip_unsupported_closing_offers(text).strip()

    @staticmethod
    def _is_planning_query(user_message: str) -> bool:
        """Detects itinerary/planning intent without over-matching words like `today`."""
        message_lower = user_message.lower()
        planning_patterns = [
            r"\bplan\b",
            r"\bplan my day\b",
            r"\bday plan\b",
            r"\bitinerary\b",
            r"\broteiro\b",
            r"\bplano\b",
            r"\bagenda\b",
            r"\bschedule\b",
            r"\bday trip\b",
            r"\bpasseio\b",
            r"\bvisitar\b",
            r"\bvisitando\b",
            r"\bplane(?:ar|ia|ie)\b",
            r"\borganiza(?:r)?\b",
            r"\bir a vários? locais\b",
            r"\bvisit multiple\b",
        ]
        return any(re.search(pattern, message_lower) for pattern in planning_patterns)

    @traceable(name="supervisor_agent", run_type="chain", tags=["sub-agent", "supervisor"])
    def route(
        self,
        user_message: str,
        language: str = "en",
        conversation_history: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes user message and returns routing decision.

        Args:
            user_message: The user's query.
            language: Language code ('en' or 'pt').
            conversation_history: Recent conversation messages for follow-up context.

        Returns:
            Dict with:
                - reasoning: Why these agents were chosen
                - agents: List of agent names to call (can be empty)
                - direct_response: Response if no agents needed (or None)
        """
        direct_override = self._direct_routing_override(user_message, language)
        if direct_override:
            return direct_override

        system_prompt = get_supervisor_prompt(language)

        messages = [SystemMessage(content=system_prompt)]

        # Inject minimal follow-up context (NOT raw messages - that confuses routing)
        if conversation_history:
            # Extract ONLY the last user query for follow-up detection
            last_user_queries = []
            for msg in reversed(conversation_history):
                if isinstance(msg, HumanMessage) and msg.content:
                    last_user_queries.append(msg.content[:150])
                    if len(last_user_queries) >= 2:
                        break
            last_user_queries.reverse()

            if last_user_queries:
                context_note = (
                    "FOLLOW-UP CONTEXT (for reference ONLY - do NOT add extra agents because of this):\n"
                    f"Previous user question(s): {' | '.join(last_user_queries)}\n"
                    "Use this ONLY to understand references like 'E amanhã?', 'E de autocarro?' etc. "
                    "Route the CURRENT query based on its OWN content. Do NOT add agents from previous topics."
                )
                messages.append(SystemMessage(content=context_note))

        messages.append(HumanMessage(content=user_message))

        # Get routing decision from LLM (with retry for Azure content filter)
        response = self._safe_llm_invoke(self.llm, messages)
        content = clean_response(response.content, _print=False)

        # Parse JSON response
        decision = parse_json_response(content)

        if decision:
            agents = decision.get("agents", [])
            reasoning = decision.get("reasoning", "")

            # Check if this is a planning query that requires weather
            is_planning_query = self._is_planning_query(user_message)

            # Force weather agent for near-future planning
            if is_planning_query and self._requires_weather_for_planning(user_message):
                if "weather" not in agents:
                    agents.append("weather")
                    reasoning += " (Added weather agent: planning for near-future date)"

            # Enforce rejection for out of scope queries even if LLM tries to answer
            reasoning_lower = reasoning.lower()
            if not agents and any(k in reasoning_lower for k in ["matemática", "math", "fora de âmbito", "out of scope", "trivia", "trivialidade"]):
                # Only override if LLM didn't provide a direct_response
                if not decision.get("direct_response"):
                    if language == "pt":
                        decision["direct_response"] = (
                            "Ups, isso fica um pouco fora da minha especialidade! 😄 "
                            "Sou o teu **Assistente Urbano de Lisboa** e estou aqui para te ajudar "
                            "a aproveitar ao máximo a Área Metropolitana de Lisboa 🏙️\n\n"
                            "Olha o que posso fazer por ti:\n\n"
                            "- 🌤️ Previsões meteorológicas e avisos em tempo real\n"
                            "- 🚇 Informação de transportes (Metro, autocarros, comboios, elétricos)\n"
                            "- 🎭 Eventos culturais e atividades\n"
                            "- 📍 Locais para visitar, restaurantes e atrações\n"
                            "- 🗺️ Planeamento de itinerários à medida\n"
                            "- 🏥 Serviços próximos (farmácias, hospitais, parques)\n"
                            "- 📚 História e cultura de Lisboa\n\n"
                            "Pergunta-me o que quiseres sobre Lisboa! 🧭"
                        )
                    else:
                        decision["direct_response"] = (
                            "Oops, that's a bit outside my expertise! 😄 "
                            "I'm your **Lisbon Urban Assistant** and I'm here to help you "
                            "make the most of the Lisbon Metropolitan Area 🏙️\n\n"
                            "Here's what I can do for you:\n\n"
                            "- 🌤️ Weather forecasts & real-time warnings\n"
                            "- 🚇 Transport info (Metro, buses, trains, trams)\n"
                            "- 🎭 Cultural events & activities\n"
                            "- 📍 Places to visit, restaurants & attractions\n"
                            "- 🗺️ Custom itinerary planning\n"
                            "- 🏥 Nearby services (pharmacies, hospitals, parks)\n"
                            "- 📚 Lisbon history & culture\n\n"
                            "Go ahead, ask me anything about Lisbon! 🧭"
                        )

            return {
                "reasoning": reasoning,
                "agents": agents,
                "direct_response": self._sanitize_direct_response(decision.get("direct_response")),
            }

        # Fallback: If JSON parsing fails, try to extract intent heuristically
        return self._fallback_routing(user_message, content, language)

    def _fallback_routing(self, user_message: str, llm_response: str, language: str = "pt") -> Dict[str, Any]:
        """
        Fallback routing when JSON parsing fails.
        Uses simple keyword matching as backup.

        Args:
            user_message: Original user query.
            llm_response: Raw LLM response that failed to parse.
            language: User language code ("pt" or "en"). Defaults to "pt".

        Returns:
            Dict with routing decision.
        """
        direct_override = self._direct_routing_override(user_message, language)
        if direct_override:
            return direct_override

        message_lower = user_message.lower()

        # 1. Check for Out-of-Scope keywords (Locations outside AML)
        # Note: AML includes Sintra, Cascais, Montijo, Setúbal, etc. - these are IN SCOPE!
        # CRITICAL: Use word boundaries to avoid false positives
        # e.g. "porto" must NOT match "aeroporto", "transporte", etc.
        forbidden_patterns = [
            r"\bporto\b",
            r"\baveiro\b",
            r"\bbraga\b",
            r"\bcoimbra\b",
            r"\bfaro\b",
            r"\balgarve\b",
            r"\bévora\b",
            r"\bmadrid\b",
            r"\bparis\b",
            r"\blondon\b",
            r"\bbarcelona\b",
            r"\bnew york\b",
            r"\btokyo\b",
            r"\broma\b",
            r"\brome\b",
        ]
        if any(re.search(pat, message_lower) for pat in forbidden_patterns):
            if language == "en":
                oos_msg = (
                    "That's a bit outside my area! 😊 "
                    "I'm your guide for the **Lisbon Metropolitan Area** 🏙️\n\n"
                    "But here's everything I can help you with:\n\n"
                    "- 🌤️ Weather forecasts and warnings\n"
                    "- 🚇 Real-time transport (Metro, buses, trains)\n"
                    "- 🎭 Cultural events and activities\n"
                    "- 📍 Places, museums and attractions\n"
                    "- 🗺️ Personalized itinerary planning\n"
                    "- 🏥 Essential services (pharmacies, hospitals, schools)\n\n"
                    "Want to explore Lisbon? Just ask! 🧭"
                )
            else:
                oos_msg = (
                    "Isso fica um pouco fora da minha área! 😊 "
                    "Sou o teu guia para a **Área Metropolitana de Lisboa** 🏙️\n\n"
                    "Mas olha tudo o que te posso ajudar:\n\n"
                    "- 🌤️ Previsão meteorológica e avisos\n"
                    "- 🚇 Transportes em tempo real (Metro, autocarros, comboios)\n"
                    "- 🎭 Eventos e atividades culturais\n"
                    "- 📍 Locais, museus e atrações\n"
                    "- 🗺️ Planeamento personalizado de itinerários\n"
                    "- 🏥 Serviços essenciais (farmácias, hospitais, escolas)\n\n"
                    "Queres explorar Lisboa? Pergunta-me! 🧭"
                )
            return {
                "reasoning": "Fallback: Detected out-of-scope location (outside AML)",
                "agents": [],
                "direct_response": self._sanitize_direct_response(oos_msg),
            }

        # 2. AML locations that ARE in scope - should trigger transport agent
        aml_keywords = [
            "sintra",
            "cascais",
            "oeiras",
            "amadora",
            "loures",
            "odivelas",
            "almada",
            "seixal",
            "barreiro",
            "montijo",
            "alcochete",
            "setúbal",
            "palmela",
            "sesimbra",
            "mafra",
            "vila franca",
        ]
        if any(loc in message_lower for loc in aml_keywords):
            # These are AML locations - use transport agent
            return {
                "reasoning": "Fallback: AML location detected - using transport agent",
                "agents": ["transport"],
                "direct_response": None,
            }

        # Weather keywords
        # Transport keywords
        transport_keywords = [
            "metro",
            "bus",
            "train",
            "carris",
            "comboio",
            "autocarro",
            "route",
            "rota",
            "como chego",
            "how to get",
            "transporte",
            "ferry",
            "barco",
            "fertagus",
            "cp",
            "frequência",
            "frequency",
            "headway",
            "intervalo",
            "de quanto em quanto",
            "how often",
        ]

        # Places/Events keywords
        places_keywords = [
            "museum",
            "restaurant",
            "park",
            "museu",
            "restaurante",
            "parque",
            "visit",
            "visitar",
            "event",
            "evento",
            "attraction",
            "atração",
            "what to do",
            "o que fazer",
            "places",
        ]

        # Resident services keywords (always → researcher)
        service_keywords = [
            "farmácia",
            "pharmacy",
            "hospital",
            "escola",
            "school",
            "biblioteca",
            "library",
            "bombeiros",
            "fire",
            "polícia",
            "police",
            "junta",
            "embaixada",
            "embassy",
            "cemitério",
            "wc",
            "sanitário",
            "toilet",
            "mercado",
            "market",
            "piscina",
            "desporto",
            "sports",
            "jardim",
            "garden",
            "creche",
            "estacionamento",
            "parking",
            "serviço",
            "service",
        ]

        agents = []

        # Check for keywords
        if self._looks_like_weather_query(message_lower):
            agents.append("weather")
        if any(kw in message_lower for kw in transport_keywords):
            agents.append("transport")
        if any(kw in message_lower for kw in places_keywords):
            agents.append("researcher")
        if any(kw in message_lower for kw in service_keywords):
            if "researcher" not in agents:
                agents.append("researcher")
        if self._is_planning_query(message_lower):
            # Itinerary needs weather + researcher + planner
            if "weather" not in agents:
                agents.append("weather")
            if "researcher" not in agents:
                agents.append("researcher")
            agents.append("planner")

        # If still no agents and not a greeting, default to researcher
        if not agents:
            agents = ["researcher"]

        return {
            "reasoning": f"Fallback routing based on keywords: {agents}",
            "agents": agents,
            "direct_response": None,
        }

    def _requires_weather_for_planning(self, user_message: str) -> bool:
        """
        Detects if user is asking to plan for today, tomorrow, this week, or next few days.
        Returns True if weather data should be mandatory for the planning.

        Args:
            user_message: The user's query.

        Returns:
            bool: True if weather should be required for planning.
        """
        message_lower = user_message.lower()

        # Patterns for immediate/near-future planning (requires weather)
        immediate_patterns = [
            # Today
            r"\bhoje\b",
            r"\btoday\b",
            r"\bfor today\b",
            r"\bpara hoje\b",
            # Tomorrow
            r"\bamanh[ãa]\b",
            r"\btomorrow\b",
            r"\bfor tomorrow\b",
            r"\bpara amanh[ãa]\b",
            # This week
            r"\besta semana\b",
            r"\bthis week\b",
            r"\bna semana\b",
            r"\bduring this week\b",
            # Next X days
            r"\bpr[óo]ximos?\s+\d+\s+dias?\b",
            r"\bnext\s+\d+\s+days?\b",
            r"\bpr[óo]ximos?\s+(?:dois|tr[êe]s|quatro|cinco|seis|sete)\s+dias?\b",
            r"\bnext\s+(?:two|three|four|five|six|seven)\s+days?\b",
            r"\b(?:dois|tr[êe]s|quatro|cinco|seis|sete)\s+dias?\b",
            r"\b(?:two|three|four|five|six|seven)\s+days?\b",
            # Day plans
            r"\bday\s+\d+\b",
            r"\b(\d+)[oa]?\s+dia\b",
            # Weekend
            r"\bweekend\b",
            r"\bfim de semana\b",
            r"\bpr[óo]ximo fim de semana\b",
            r"\bnext weekend\b",
            # Now/immediate
            r"\bagora\b",
            r"\bnow\b",
            r"\bcurrently\b",
            r"\batualmente\b",
        ]

        # Check if any immediate pattern matches
        for pattern in immediate_patterns:
            if re.search(pattern, message_lower):
                return True

        return False

    def format_agent_outputs(self, agent_outputs: Dict[str, str]) -> str:
        """
        Formats outputs from multiple agents into a combined context string.

        Args:
            agent_outputs: Dict mapping agent names to their outputs.

        Returns:
            str: Formatted context for the planner or final response.
        """
        if not agent_outputs:
            return ""

        sections = []

        if "weather" in agent_outputs:
            sections.append(f"## 🌤️ Weather Information\n{agent_outputs['weather']}")

        if "transport" in agent_outputs:
            sections.append(
                f"## 🚇 Transport Information\n{agent_outputs['transport']}"
            )

        if "researcher" in agent_outputs:
            sections.append(f"## 🔍 Places & Events\n{agent_outputs['researcher']}")

        return "\n\n---\n\n".join(sections)


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Supervisor Agent Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")

    passed = 0
    failed = 0

    try:
        supervisor = SupervisorAgent()
        print(
            f"\n\033[1m✅ Supervisor initialized:\033[0m {supervisor.get_model_info()}"
        )

        # =================================================================
        # ORIGINAL ROUTING TESTS
        # =================================================================
        print("\n\033[1m📋 Section 1: General Routing\033[0m")
        print("-" * 50)

        test_queries = [
            "Hello!",
            "What's the weather like?",
            "How do I get to Belém?",
            "Recommend some museums",
            "Plan my day visiting museums and considering the weather",
            "Quanto é 2+2?",
            "Que sitios posso visitar no Porto?",
            "Como está o tempo em Aveiro?",
        ]

        for query in test_queries:
            print(f"\n\033[1m📝 Query:\033[0m {query}")
            decision = supervisor.route(query)
            print(f"   \033[1mAgents:\033[0m {decision['agents']}")
            print(f"   \033[1mReason:\033[0m {decision['reasoning']}")
            if decision["direct_response"]:
                print(f"   \033[1mDirect:\033[0m {decision['direct_response']}")

        # =================================================================
        # SERVICE KEYWORD ROUTING TESTS
        # =================================================================
        print("\n\033[1m📋 Section 2: Resident Service Routing\033[0m")
        print("-" * 50)

        service_tests = [
            ("Onde fica a farmácia mais próxima?", "researcher", "pharmacy → researcher"),
            ("Where is the nearest hospital?", "researcher", "hospital → researcher"),
            ("Há alguma biblioteca perto de mim?", "researcher", "library → researcher"),
            ("I need a police station nearby", "researcher", "police → researcher"),
            ("Onde posso estacionar perto do Rossio?", "researcher", "parking → researcher"),
            ("Preciso de encontrar uma escola para o meu filho", "researcher", "school → researcher"),
            ("Where is the nearest WC?", "researcher", "wc/toilet → researcher"),
            ("Quero encontrar um mercado", "researcher", "market → researcher"),
            ("Onde ficam os bombeiros?", "researcher", "fire station → researcher"),
            ("Há piscinas municipais abertas?", "researcher", "sports/pool → researcher"),
        ]

        for query, expected_agent, description in service_tests:
            decision = supervisor._fallback_routing(query, "")
            agents = decision["agents"]
            if expected_agent in agents:
                passed += 1
                print(f"  \033[1;32m✅ PASS\033[0m: {description}")
                print(f"      Query: {query}")
                print(f"      Agents: {agents}")
            else:
                failed += 1
                print(f"  \033[1;31m❌ FAIL\033[0m: {description}")
                print(f"      Query: {query}")
                print(f"      Expected '{expected_agent}' in {agents}")

        # =================================================================
        # FREQUENCY KEYWORD ROUTING TESTS
        # =================================================================
        print("\n\033[1m📋 Section 3: Frequency/Headway Routing\033[0m")
        print("-" * 50)

        frequency_tests = [
            ("How often does the metro come?", "transport", "frequency EN → transport"),
            ("De quanto em quanto tempo passa o 28E?", "transport", "frequency PT → transport"),
            ("What's the headway on the Sintra line?", "transport", "headway → transport"),
            ("Qual a frequência do comboio para Cascais?", "transport", "frequência → transport"),
            ("What's the interval between buses?", "transport", "intervalo → transport"),
        ]

        for query, expected_agent, description in frequency_tests:
            decision = supervisor._fallback_routing(query, "")
            agents = decision["agents"]
            if expected_agent in agents:
                passed += 1
                print(f"  \033[1;32m✅ PASS\033[0m: {description}")
                print(f"      Agents: {agents}")
            else:
                failed += 1
                print(f"  \033[1;31m❌ FAIL\033[0m: {description}")
                print(f"      Expected '{expected_agent}' in {agents}")

        # =================================================================
        # OUT-OF-SCOPE FALLBACK TESTS
        # =================================================================
        print("\n\033[1m📋 Section 4: Out-of-Scope Fallback\033[0m")
        print("-" * 50)

        oos_tests = [
            ("What is the capital of Japan?", "General OOS query"),
            ("Como se diz obrigado em mandarim?", "Language/trivia OOS"),
            ("Bom dia!", "Greeting"),
        ]

        for query, description in oos_tests:
            decision = supervisor._fallback_routing(query, "")
            agents = decision["agents"]
            print(f"  \033[1m📝\033[0m {description}: \"{query[:40]}\"")
            print(f"      Agents: {agents} | Direct: {'Yes' if decision['direct_response'] else 'No'}")

        # =================================================================
        # SUMMARY
        # =================================================================
        total = passed + failed
        print("\n" + "=" * 60)
        print("\033[1m📊 SUPERVISOR TEST SUMMARY\033[0m")
        print("=" * 60)
        print(f"\033[1;32m✅ Passed: {passed}/{total}\033[0m")
        if failed > 0:
            print(f"\033[1;31m❌ Failed: {failed}/{total}\033[0m")
        else:
            print("\033[1;32m🎉 ALL FALLBACK ROUTING TESTS PASSED!\033[0m")
        print("=" * 60)

        print("\n\033[1;32m✅ Supervisor agent working!\033[0m")

    except Exception as e:
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
        import traceback
        traceback.print_exc()
