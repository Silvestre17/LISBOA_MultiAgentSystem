# ==========================================================================
# Master Thesis - Follow-Up Interpretation Prompt
#   - André Filipe Gomes Silvestre, 20240502
#
#   Reads the latest user message against the recent conversation and
#   rewrites it as one self-contained request, so routing, workers, and the
#   planner never have to resolve references to earlier turns themselves.
#   A comparison between items of an earlier answer is answered from that
#   answer alone.
# ==========================================================================

from langchain_core.messages import HumanMessage, SystemMessage

from tools.utils import lisbon_now

FOLLOW_UP_KINDS = ("new", "plan_change", "follow_up", "more_results", "recall", "compare", "clarify")

# The examples follow the answer language: a Portuguese example question
# ("Qual deles: ...?") is otherwise copied into English answers.
_EXAMPLES = {
    "pt": """Examples (after "Estou no Saldanha. Planeia a manhã de sábado no Parque das Nações de metro" was answered with the Oceanário, the Pavilhão do Conhecimento, and the Jardim Garcia de Orta):
- "Quero começar pelo jardim" -> plan_change, "Planeia a manhã de sábado no Parque das Nações, a partir do Saldanha e de metro, começando a visita pelo Jardim Garcia de Orta e seguindo depois para o Oceanário e o Pavilhão do Conhecimento."
- "E para voltar no fim?" -> follow_up, "Como vou do Pavilhão do Conhecimento para o Saldanha, de metro, no sábado ao fim da manhã?"
- "Esse museu tem bilhetes online?" -> follow_up, "O Pavilhão do Conhecimento, no Parque das Nações, tem bilhetes online?"
- "E se chover?" -> plan_change, "Planeia a manhã de sábado no Parque das Nações, a partir do Saldanha e de metro, só com visitas interiores ou cobertas, para um dia de chuva, mantendo o Oceanário e o Pavilhão do Conhecimento."
- "Qual destes é melhor para crianças?" -> compare, "Qual é o melhor para crianças: o Oceanário, o Pavilhão do Conhecimento ou o Jardim Garcia de Orta?"
- "Como lá chego?" -> clarify, answer "Para qual deles: o Oceanário, o Pavilhão do Conhecimento ou o Jardim Garcia de Orta?\"""",
    "en": """Examples (after "I'm at Saldanha. Plan Saturday morning in Parque das Nações by metro" was answered with the Oceanário, the Pavilhão do Conhecimento, and the Jardim Garcia de Orta):
- "I want to start with the garden" -> plan_change, "Plan Saturday morning in Parque das Nações, from Saldanha by metro, starting with the Jardim Garcia de Orta and then the Oceanário and the Pavilhão do Conhecimento."
- "And to get back at the end?" -> follow_up, "How do I get from the Pavilhão do Conhecimento to Saldanha, by metro, on Saturday late morning?"
- "Does that museum sell tickets online?" -> follow_up, "Does the Pavilhão do Conhecimento, in Parque das Nações, sell tickets online?"
- "What if it rains?" -> plan_change, "Plan Saturday morning in Parque das Nações, from Saldanha by metro, with indoor or covered visits only for a rainy day, keeping the Oceanário and the Pavilhão do Conhecimento."
- "Which of these is better for kids?" -> compare, "Which is best for children: the Oceanário, the Pavilhão do Conhecimento, or the Jardim Garcia de Orta?"
- "How do I get there?" -> clarify, answer "Which one: the Oceanário, the Pavilhão do Conhecimento, or the Jardim Garcia de Orta?\"""",
}


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
    examples = _EXAMPLES["pt" if language == "pt" else "en"]
    system = f"""
You read the latest message of a conversation with LISBOA, a Lisbon travel assistant, and return ONE JSON object: {{"kind": "...", "request": "...", "answer": "..."}}. No prose, no Markdown.

kind:
- "new": the message is complete on its own and does not depend on earlier turns: a new subject (another place, route, weather, event, or plan request with its own area or destination), or a change of topic.
- "plan_change": the message changes the latest itinerary: where it starts or which stop comes first, where it ends, the order, adding, removing, or swapping a stop or meal, another day or part of the day, a weather adaptation ("e se chover?", "what if it rains?"), the transport mode, the budget, the pace, or the audience.
- "follow_up": the message continues an earlier answer without changing the itinerary: another mode or the trip back for the last route, the next departure, how to reach a stop of the itinerary or an item of a list ("o primeiro", "esse museu", "lá", "no fim"), the weather on another day for the same place, events or places "there", or the answer to a question the assistant asked (an origin, a date, an option).
- "more_results": the user asks for more options of the latest list ("mais opções", "more", "outros").
- "recall": the user asks what an earlier answer said ("qual era o restaurante?", "a que horas era o almoço?") and the earlier answers contain it.
- "compare": the user asks LISBOA to choose, rank, or compare items of an earlier answer by a criterion ("qual destes é mais barato?", "qual deles é melhor para crianças?", "which of these is better for kids?", "which one opens earliest?", "o mais bem avaliado?"). The user wants an answer, so it is never "clarify".
- "clarify": the message asks about one earlier item (how to get there, its hours, its tickets) through a reference ("esse", "that one", "esse museu", "o restaurante", "lá") but several items of the earlier answers fit and the message does not say which.

request: write it in {language_name}.
- "new": copy the message unchanged.
- "follow_up" and "more_results": one complete request that an assistant with no access to this conversation could answer. Replace every reference with the exact names from the earlier answers ("lá" -> the place, "o primeiro" -> the first item's name, "no fim" -> the last stop of the itinerary, "para voltar" -> the route in the opposite direction), and keep the constraints still in force (origin, destination, day, transport mode, area, category, and preferences the user stated earlier, such as diet, budget, or accessibility). The reference words themselves ("lá", "no fim", "esse", "there", "at the end") do not stay in the request. "Lá" in a request for events, places, or weather is the area or town of the place it refers to ("eventos lá" after a restaurant in Cascais -> "eventos em Cascais"); "esse museu" or "that restaurant" is the one item of that type in the earlier answer.
- "plan_change": one complete itinerary request: the start point, the area, the day and part of the day, the transport mode, the other constraints the user gave (earlier requests included), and the stops of the latest itinerary that stay, in visiting order, with the change applied (for "start at the Padrão dos Descobrimentos": "começando a visita pelo Padrão dos Descobrimentos e seguindo depois para ..."). Keep the earlier day and part of the day unless the user changes them.
- "compare": one complete question that names every item being compared, exactly as the earlier answer writes them, and the criterion ("Qual é o mais barato: o Museu de Marinha, o MAAT ou o Museu Nacional dos Coches?").
- "recall" and "clarify": copy the message unchanged.
For every kind except "new", "recall", and "clarify", the request must stand alone: never copy the message as written when it leaves out the start point, the area, the day, or the place it refers to.
A route request names the origin and the destination first, as plain place names, and the mode and the day after them: "Como vou de <origem> para <destino>, de <modo>, <dia>?", "How do I get from <origin> to <destination>, by <mode>, <day>?".
The transport mode is the one the user asked for ("transportes públicos", "de metro", "by train"); a line or mode that appears only in the assistant's routes (elétrico 15E, a bus number) is not a constraint and never goes into the request.
In a rewritten request, write a day the latest message names ("amanhã", "no domingo", "on Friday") as its weekday and date, counted from today: a weekday is the next one after today, unless the message says today ("hoje", "today"). When the latest message names no day, keep the day of the itinerary or route it continues, copied as the conversation writes it; never work out another one.

answer: in {language_name}. For "recall", one or two sentences that answer from the earlier answers alone, or "" when they do not show it. For "clarify", one short question that names the items that fit, as in the clarify example below. Otherwise "".

Use only names and facts that appear in the conversation, and write each name as the user or the earlier answer wrote it, never translated ("Padrão dos Descobrimentos" stays "Padrão dos Descobrimentos" in a request in any language). Never add a place, time, or line that it does not show.

{examples}
""".strip()
    today = lisbon_now().strftime("%A, %d %B %Y")
    human = f"# Today\n{today}\n\n# Conversation so far\n{conversation}\n\n# Latest message\n{user_message}"
    return [SystemMessage(content=system), HumanMessage(content=human)]


def build_comparison_messages(user_message: str, question: str, language: str, earlier_answer: str) -> list:
    """Build the messages that answer a comparison between items of an earlier answer.

    Args:
        user_message: The user's message as written.
        question: The standalone comparison question (items and criterion).
        language: Answer language of the turn (``pt`` or ``en``).
        earlier_answer: The earlier answer that lists the compared items.

    Returns:
        LangChain messages whose answer is Markdown without a source footer.
    """
    is_pt = language == "pt"
    language_name = "European Portuguese (PT-PT)" if is_pt else "English"
    direct_label = "Resposta direta" if is_pt else "Direct answer"
    limitation_label = "Limitação" if is_pt else "Limitation"
    # Sample phrases follow the answer language, so none is copied across languages.
    no_price = "sem preço indicado" if is_pt else "no price is listed"
    partial_example = (
        "só o Museu A indica preço de bilhete (adulto 7 €); os outros não indicam preço, "
        "por isso não é possível dizer qual é o mais barato"
        if is_pt
        else "only Museum A lists a ticket price (adult 7 €); the others list none, "
        "so the cheapest cannot be confirmed"
    )
    system = f"""
You are LISBOA, a Lisbon travel assistant. The user asks you to choose, rank, or compare items that your earlier answer listed. Answer from that earlier answer alone, in {language_name}, as Markdown in this shape:

### <one emoji> **<short title naming the criterion>**

✅ **{direct_label}:** <the item or items that best meet the criterion, by name, and the fact from the earlier answer that decides it; when the deciding value is missing for some items, say that the winner cannot be confirmed, as in "{partial_example}">

---

- **<item name>:** <the facts of the earlier answer that bear on the criterion, in one short line>
(one bullet per compared item, the best first)

⚠️ **{limitation_label}:** <only when the earlier answer does not show the criterion for some items: what is missing and what the user should confirm with the venue>

Rules:
- Compare the items the user's words refer to: "these", "destes", "deles", "which one" mean every item of the earlier list, even when the rewritten question names fewer.
- Always write the direct answer line, also when the verdict is that nothing can be confirmed.
- Use only the facts the earlier answer shows (descriptions, categories, prices, opening hours, ratings, features, addresses). Never add a fact, a place, or a price from general knowledge.
- Anything "with Lisboa Card" ("Free with Lisboa Card", "Gratuito com Lisboa Card", "20% com Lisboa Card") is a benefit for card holders: it is not free entry, not a ticket price, and never decides "cheapest". Only ticket prices and plain free entry ("Free", "Gratuito") do.
- A criterion decided by a value (ticket price, rating, opening or closing time) needs that value for every item. When only some items show it, the direct answer compares those and says the winner cannot be confirmed ("{partial_example}"); it never calls one the cheapest, best rated, or earliest of all. When no item shows it, the direct answer says that no item shows it.
- An audience criterion (children, families, accessibility, seniors) is rarely stated as such: rank by the signals the earlier answer does show (child or family tickets, interactive or hands-on content in the description, step-free access in the features), name those signals, and add the limitation that suitability is not confirmed.
- Write every item name exactly as the earlier answer writes it, never translated or partly translated.
- Bold only item names. No quotation marks, no links, no source line, no closing offer.
- Speak about the places, not about this conversation: "{no_price}", never a mention of an earlier answer; no words about tools, data sources, or records.
""".strip()
    human = f"# Earlier answer\n{earlier_answer}\n\n# User's words\n{user_message}\n\n# Question\n{question}"
    return [SystemMessage(content=system), HumanMessage(content=human)]
