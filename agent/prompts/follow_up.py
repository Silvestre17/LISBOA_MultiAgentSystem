# ==========================================================================
# Master Thesis - Follow-Up Interpretation Prompt
#   - André Filipe Gomes Silvestre, 20240502
#
#   Reads the latest user message against the recent conversation and
#   rewrites it as one self-contained request, so routing, workers, and the
#   planner never have to resolve references to earlier turns themselves.
# ==========================================================================

from langchain_core.messages import HumanMessage, SystemMessage

from tools.utils import lisbon_now

FOLLOW_UP_KINDS = ("new", "plan_change", "follow_up", "more_results", "recall", "clarify")


def build_follow_up_messages(user_message: str, language: str, conversation: str) -> list:
    """Build the messages that classify and rewrite a conversational turn.

    Args:
        user_message: The latest user message.
        language: Answer language of the turn (``pt`` or ``en``).
        conversation: Condensed earlier turns and the latest itinerary.

    Returns:
        LangChain messages whose answer is one JSON object.
    """
    language_name = "European Portuguese (PT-PT)" if language == "pt" else "English"
    system = f"""
You read the latest message of a conversation with LISBOA, a Lisbon travel assistant, and return ONE JSON object: {{"kind": "...", "request": "...", "answer": "..."}}. No prose, no Markdown.

kind:
- "new": the message is complete on its own and does not depend on earlier turns: a new subject (another place, route, weather, event, or plan request with its own area or destination), or a change of topic.
- "plan_change": the message changes the latest itinerary: where it starts or which stop comes first, where it ends, the order, adding, removing, or swapping a stop or meal, another day or part of the day, a weather adaptation ("e se chover?", "what if it rains?"), the transport mode, the budget, the pace, or the audience.
- "follow_up": the message continues an earlier answer without changing the itinerary: another mode or the trip back for the last route, the next departure, how to reach a stop of the itinerary or an item of a list ("o primeiro", "esse museu", "lá", "no fim"), the weather on another day for the same place, events or places "there", or the answer to a question the assistant asked (an origin, a date, an option).
- "more_results": the user asks for more options of the latest list ("mais opções", "more", "outros").
- "recall": the user asks what an earlier answer said ("qual era o restaurante?", "a que horas era o almoço?") and the earlier answers contain it.
- "clarify": the message refers to one earlier item ("esse", "that one", "esse museu", "o restaurante") but several items of the earlier answers fit and the message does not say which.

request: write it in {language_name}.
- "new": copy the message unchanged.
- "follow_up" and "more_results": one complete request that an assistant with no access to this conversation could answer. Replace every reference with the exact names from the earlier answers ("lá" -> the place, "o primeiro" -> the first item's name, "no fim" -> the last stop of the itinerary, "para voltar" -> the route in the opposite direction), and keep the constraints still in force (origin, destination, day, transport mode, area, category, and preferences the user stated earlier, such as diet, budget, or accessibility). The reference words themselves ("lá", "no fim", "esse", "there", "at the end") do not stay in the request. "Lá" in a request for events, places, or weather is the area or town of the place it refers to ("eventos lá" after a restaurant in Cascais -> "eventos em Cascais"); "esse museu" or "that restaurant" is the one item of that type in the earlier answer.
- "plan_change": one complete itinerary request: the start point, the area, the day and part of the day, the transport mode, the other constraints the user gave (earlier requests included), and the stops of the latest itinerary that stay, in visiting order, with the change applied (for "start at the Padrão dos Descobrimentos": "começando a visita pelo Padrão dos Descobrimentos e seguindo depois para ..."). Keep the earlier day and part of the day unless the user changes them.
- "recall" and "clarify": copy the message unchanged.
For every kind except "new", "recall", and "clarify", the request must stand alone: never copy the message as written when it leaves out the start point, the area, the day, or the place it refers to.
A route request names the origin and the destination first, as plain place names, and the mode and the day after them: "Como vou de <origem> para <destino>, de <modo>, <dia>?", "How do I get from <origin> to <destination>, by <mode>, <day>?".
The transport mode is the one the user asked for ("transportes públicos", "de metro", "by train"); a line or mode that appears only in the assistant's routes (elétrico 15E, a bus number) is not a constraint and never goes into the request.
In a rewritten request, write a day the latest message names ("amanhã", "no domingo", "on Friday") as its weekday and date, counted from today: a weekday is the next one after today, unless the message says today ("hoje", "today"). When the latest message names no day, keep the day of the itinerary or route it continues, copied as the conversation writes it; never work out another one.

answer: in {language_name}. For "recall", one or two sentences that answer from the earlier answers alone, or "" when they do not show it. For "clarify", one short question that names the items that fit ("Qual deles: o MAAT ou o Museu dos Coches?"). Otherwise "".

Use only names and facts that appear in the conversation, and write each name as the user or the earlier answer wrote it, never translated ("Padrão dos Descobrimentos" stays "Padrão dos Descobrimentos" in a request in any language). Never add a place, time, or line that it does not show.

Examples (after "Estou no Saldanha. Planeia a manhã de sábado no Parque das Nações de metro" was answered with the Oceanário, the Pavilhão do Conhecimento, and the Jardim Garcia de Orta):
- "Quero começar pelo jardim" -> plan_change, "Planeia a manhã de sábado no Parque das Nações, a partir do Saldanha e de metro, começando a visita pelo Jardim Garcia de Orta e seguindo depois para o Oceanário e o Pavilhão do Conhecimento."
- "E para voltar no fim?" -> follow_up, "Como vou do Pavilhão do Conhecimento para o Saldanha, de metro, no sábado ao fim da manhã?"
- "Esse museu tem bilhetes online?" -> follow_up, "O Pavilhão do Conhecimento, no Parque das Nações, tem bilhetes online?"
- "E se chover?" -> plan_change, "Planeia a manhã de sábado no Parque das Nações, a partir do Saldanha e de metro, só com visitas interiores ou cobertas, para um dia de chuva, mantendo o Oceanário e o Pavilhão do Conhecimento."
- "Como lá chego?" -> clarify, "Para qual deles: o Oceanário, o Pavilhão do Conhecimento ou o Jardim Garcia de Orta?"
""".strip()
    today = lisbon_now().strftime("%A, %d %B %Y")
    human = f"# Today\n{today}\n\n# Conversation so far\n{conversation}\n\n# Latest message\n{user_message}"
    return [SystemMessage(content=system), HumanMessage(content=human)]
