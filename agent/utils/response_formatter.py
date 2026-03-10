# ==========================================================================
# Master Thesis - Response Formatter
#   - André Filipe Gomes Silvestre, 20240502
#
#   Post-processing pipeline to ensure LLM responses render cleanly
#   in Streamlit's st.markdown(). Includes link normalization,
#   metro terminology cleanup, header and bullet normalization,
#   response-title helpers, and final formatting for consistent visual quality.
# ==========================================================================

import re
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

_SOURCE_LINE_RE = re.compile(r'^(?:[-*•]\s*)?(?:📌\s*)?(?:\*\*)?(?:Fonte|Source)(?:\*\*)?:.*$', re.IGNORECASE)
_PT_LANGUAGE_HINTS_RE = re.compile(
    r"\b(olá|ola|bom dia|boa tarde|boa noite|como|quero|preciso|museu|museus|evento|eventos|hoje|amanhã|amanha|previsão|tempo|locais|morada|fonte|autocarro|comboio|bairro)\b",
    re.IGNORECASE,
)
_EVENT_HINTS_RE = re.compile(
    r"\b(event|events|evento|eventos|concert|concerto|festival|exhibition|exposição|exposicao|show|espetáculo|espetaculo|what's on|o que há|o que ha)\b",
    re.IGNORECASE,
)
_PLACE_HINTS_RE = re.compile(
    r"\b(place|places|museum|museums|museu|museus|attraction|attractions|atração|atrações|atracao|atracoes|restaurant|restaurants|restaurante|restaurantes|monument|monuments|local|locais)\b",
    re.IGNORECASE,
)
_ACCESSIBILITY_QUERY_RE = re.compile(
    r"\b(wheelchair|accessible|accessibility|step[- ]?free|reduced mobility|cadeira de rodas|acess[ií]vel|mobilidade reduzida)\b",
    re.IGNORECASE,
)
_ACCESSIBILITY_CLAIM_RE = re.compile(
    r"\b(wheelchair|accessible|accessibility|step[- ]?free|elevator|lift|ramp|adapted toilet|accessible restroom|cadeira de rodas|acess[ií]vel|elevador|rampa|wc adaptado)\b",
    re.IGNORECASE,
)
_INLINE_OFFER_RE = re.compile(
    r"(?:\s+|^)(?:If you want(?:,)?|If you['’]d like(?:,)?|Would you like me to|Let me know if|I can also|I can help(?: you)?|I can bring|I can fetch|I can filter|I can get updated|Se quiser(?:es)?(?:,)?|Se preferir(?:,)?|Posso também|Posso tambem|Posso detalhar|Posso filtrar|Posso trazer|Posso ver|Posso verificar|Posso procurar|Quer que eu)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


def infer_response_language(
    user_query: str = "",
    context_text: str = "",
    default: str = "en",
) -> str:
    """
    Infers the preferred response language from the user query first and the
    existing text second.

    Args:
        user_query: Original user query, if available.
        context_text: Response text or context hints.
        default: Fallback language code.

    Returns:
        str: `pt` or `en`.
    """
    if user_query:
        return "pt" if _PT_LANGUAGE_HINTS_RE.search(user_query) else (default if default in {"pt", "en"} else "en")

    combined = context_text.strip()
    if not combined:
        return default if default in {"pt", "en"} else "en"

    if _PT_LANGUAGE_HINTS_RE.search(combined):
        return "pt"

    return default if default in {"pt", "en"} else "en"


def has_source_line(text: str) -> bool:
    """Returns whether the text already contains a source line."""
    return bool(text and _SOURCE_LINE_RE.search(text))


def strip_unsupported_closing_offers(text: str) -> str:
    """
    Removes closing notes or offers that imply capabilities the system does not
    support, such as filtering extra data, fetching updated prices, reminders,
    or other post-answer actions.

    Args:
        text: Raw model response text.

    Returns:
        str: Text without unsupported closing offers.
    """
    if not text:
        return text

    prefix = r'^(?:[-*•]\s*)?(?:[⚠️💡📌🌤️🌧️🚇🎭📍]\s*)?(?:\*\*\s*)?'

    offer_patterns = [
        re.compile(prefix + r'(?:observa(?:ç|c)ão|observacao|observation|nota|note)(?:\s*\*\*)?\s*:', re.IGNORECASE),
        re.compile(
            prefix + r"(?:if you want(?:,)?|if you['’]d like(?:,)?|would you like me to|let me know if|i can also|i can help(?: you)?|i can bring|i can fetch|i can filter|i can get updated|se quiser(?:es)?(?:,)?|se preferir(?:,)?|posso também|posso tambem|posso detalhar|posso filtrar|posso trazer|posso ver|posso verificar|posso procurar|quer que eu)(?:\b|:)",
            re.IGNORECASE,
        ),
    ]

    cleaned_lines = []
    skipping_offer_block = False
    for line in text.splitlines():
        stripped = line.strip()

        if skipping_offer_block:
            if not stripped:
                skipping_offer_block = False
                continue
            if _SOURCE_LINE_RE.match(stripped):
                skipping_offer_block = False
            elif stripped.startswith(("-", "*", "•")):
                continue
            else:
                skipping_offer_block = False

        if any(pattern.match(stripped) for pattern in offer_patterns):
            skipping_offer_block = True
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = _INLINE_OFFER_RE.sub("", cleaned)
    return clean_newlines(cleaned).strip()


def _replace_source_line(
    text: str,
    replacement: str,
    predicate=None,
) -> str:
    """
    Replaces matching source lines or appends a new one if none match.

    Args:
        text: Existing response text.
        replacement: Canonical source line.
        predicate: Callable that decides whether an existing line should be
            replaced. Defaults to matching any source line.

    Returns:
        str: Updated response text.
    """
    if not text:
        return replacement.strip()

    matcher = predicate or (lambda line: bool(_SOURCE_LINE_RE.match(line.strip())))
    lines = text.splitlines()
    result = []
    replaced = False

    for line in lines:
        if matcher(line):
            if not replaced:
                result.append(replacement)
                replaced = True
            continue
        result.append(line)

    while result and not result[-1].strip():
        result.pop()

    if not replaced:
        if result:
            result.append("")
        result.append(replacement)

    return "\n".join(result).strip()


def extract_update_time(text: str) -> Optional[str]:
    """Extracts an HH:MM update timestamp from tool text when available."""
    if not text:
        return None

    patterns = [
        r"(?:📅|🔄)?\s*(?:\*\*)?(?:Updated|Atualizado)(?:\*\*)?\s*:\s*(\d{2}:\d{2})\b",
        r"(?:📅|🔄)?\s*(?:\*\*)?(?:Updated|Atualizado)(?:\*\*)?\s*:\s*\d{4}-\d{2}-\d{2}[T ](\d{2}:\d{2})(?::\d{2})?\b",
        r"\bdataUpdate\s*[:=]\s*['\"]?\d{4}-\d{2}-\d{2}[T ](\d{2}:\d{2})(?::\d{2})?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def strip_weather_update_lines(text: str) -> str:
    """Removes raw weather update lines once the timestamp has been captured."""
    if not text:
        return text

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^(?:📅|🔄)?\s*(?:\*\*)?(?:Updated|Atualizado)(?:\*\*)?\s*:", stripped, flags=re.IGNORECASE):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def canonicalize_weather_source_line(
    text: str,
    language: str = "en",
    timestamp: Optional[str] = None,
) -> str:
    """
    Ensures a single canonical IPMA source line with the same structure used by
    the transport responses.

    Args:
        text: Existing response text.
        language: Preferred response language.
        timestamp: Optional HH:MM override.

    Returns:
        str: Updated response with a canonical weather source line.
    """
    now = timestamp or extract_update_time(text) or datetime.now().strftime("%H:%M")
    if language == "pt":
        replacement = (
            f"📌 **Fonte:** Dados do [*IPMA*](https://www.ipma.pt) | **Atualizado:** {now}"
        )
    else:
        replacement = (
            f"📌 **Source:** Data from [*IPMA*](https://www.ipma.pt/en/) | **Updated:** {now}"
        )

    return _replace_source_line(text, replacement)


def canonicalize_weather_terms(text: str, language: str = "en") -> str:
    """Normalizes common weather labels to the requested display language."""
    if not text or language not in {"en", "pt"}:
        return text

    if language == "en":
        replacements = [
            (r"\*\*Avisos Meteorológicos:\*\*", "**Active Warnings:**"),
            (r"\*\*Dicas Práticas\*\*", "**Practical Tips**"),
            (r"\bAs condições meteorológicas são normais\b", "Weather conditions are normal"),
            (r"\*\*Temperatura\*\*:", "**Temperature**:"),
            (r"\*\*Condições\*\*:", "**Conditions**:"),
            (r"\*\*(?:Precipitação|Chuva)\*\*:", "**Rain**:"),
            (r"\*\*Vento\*\*:", "**Wind**:"),
            (r"\*\*Agitação Marítima\*\*", "**Rough Sea**"),
            (r"\bPeríodo:", "Period:"),
            (r"\bOndas de\b", "Waves of"),
            (r"\bsem precipitação\b", "no precipitation"),
            (r"\bsem avisos meteorológicos ativos\b", "no active weather warnings"),
            (r"\bNoroeste\b", "Northwest"),
            (r"\bNordeste\b", "Northeast"),
            (r"\bSudoeste\b", "Southwest"),
            (r"\bSudeste\b", "Southeast"),
            (r"\bNorte\b", "North"),
            (r"\bSul\b", "South"),
            (r"\bOeste\b", "West"),
            (r"\bLeste\b", "East"),
            (r"\bmoderado\b", "moderate"),
            (r"\bfraco\b", "light"),
            (r"\bforte\b", "strong"),
            (r"\bBom dia para atividades ao ar livre\b", "Good conditions for outdoor activities"),
        ]
    else:
        replacements = [
            (r"Lisbon Weather Summary", "Resumo Meteorológico de Lisboa"),
            (r"\bNo active weather warnings for area '([^']+)'\.", r"Sem avisos meteorológicos ativos para a área '\1'."),
            (r"\bNo active weather warnings\b", "Sem avisos meteorológicos ativos"),
            (r"\bNo Avisos Meteorológicos\b", "Sem avisos meteorológicos ativos"),
            (r"\bWeather conditions are normal\b", "As condições meteorológicas são normais"),
            (r"Active Weather Warnings", "Avisos Meteorológicos"),
            (r"Weather Forecast for Lisbon", "Previsão do Tempo para Lisboa"),
            (r"\bRain probability\b", "Probabilidade de chuva"),
            (r"\bUpdated\b", "Atualizado"),
            (r"\bToday\b", "Hoje"),
            (r"\*\*Level\*\*:", "**Nível**:"),
            (r"\bBe aware\b", "Tenha atenção"),
            (r"\bPeriod\b", "Período"),
            (r"\bRough sea\b", "Agitação marítima"),
            (r"\bMonday\b", "Segunda-feira"),
            (r"\bTuesday\b", "Terça-feira"),
            (r"\bWednesday\b", "Quarta-feira"),
            (r"\bThursday\b", "Quinta-feira"),
            (r"\bFriday\b", "Sexta-feira"),
            (r"\bSaturday\b", "Sábado"),
            (r"\bSunday\b", "Domingo"),
            (r"\bClear sky\b", "Céu limpo"),
            (r"\bSunny intervals\b", "Períodos de céu limpo"),
            (r"\bLight rain\b", "Aguaceiros leves"),
            (r"\bLight showers/rain\b", "Chuviscos/chuva fraca"),
            (r"\bHeavy showers/rain\b", "Aguaceiros/chuva forte"),
            (r"\bShowers/rain\b", "Aguaceiros/chuva"),
            (r"\bRain/showers\b", "Chuva/aguaceiros"),
            (r"\bIntermittent rain\b", "Chuva intermitente"),
            (r"\bIntermittent light rain\b", "Chuva fraca intermitente"),
            (r"\bIntermittent heavy rain\b", "Chuva forte intermitente"),
            (r"\bDrizzle\b", "Chuvisco"),
            (r"\bMist\b", "Bruma"),
            (r"\bFog\b", "Nevoeiro"),
            (r"\bPartly cloudy\b", "Parcialmente nublado"),
            (r"\bVery likely\b", "Muito provável"),
            (r"\bVery unlikely\b", "Muito improvável"),
            (r"\bPossible\b", "Possível"),
            (r"\bLikely\b", "Provável"),
            (r"\bUnlikely\b", "Improvável"),
            (r"\bNo rain expected\b", "sem precipitação"),
            (r"\*\*Temperature\*\*:", "**Temperatura**:"),
            (r"\*\*Conditions\*\*:", "**Condições**:"),
            (r"\*\*Rain\*\*:", "**Chuva**:"),
            (r"\*\*Wind\*\*:", "**Vento**:"),
            (r"(\d+(?:\.\d+)?°C)\s+to\s+(\d+(?:\.\d+)?°C)", r"\1 a \2"),
            (r"\bIntensity(?=\s*:)\b", "intensidade"),
            (r"\bNorthwest\b", "Noroeste"),
            (r"\bNortheast\b", "Nordeste"),
            (r"\bSouthwest\b", "Sudoeste"),
            (r"\bSoutheast\b", "Sudeste"),
            (r"\bNorth\b", "Norte"),
            (r"\bSouth\b", "Sul"),
            (r"\bWest\b", "Oeste"),
            (r"\bEast\b", "Leste"),
            (r"\bWeak\b", "fraca"),
            (r"\bModerate\b", "moderado"),
            (r"\bStrong\b", "forte"),
        ]

    normalized = text
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


def structure_weather_markdown(text: str) -> str:
    """Converts flat weather tool text into nested markdown lists for cleaner rendering."""
    if not text:
        return text

    weekday_tokens = (
        "segunda-feira",
        "terça-feira",
        "quarta-feira",
        "quinta-feira",
        "sexta-feira",
        "sábado",
        "domingo",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    )
    detail_prefixes = ("🌡️", "🌤️", "💧", "💨", "📝", "Level:", "Nível:")
    day_emojis = ("☀️", "☁️", "🌧️", "⛈️", "🌫️", "❄️", "🌦️")
    section_markers = (
        "Resumo Meteorológico de Lisboa",
        "Lisbon Weather Summary",
        "Previsão do Tempo para Lisboa",
        "Weather Forecast for Lisbon",
        "Avisos Meteorológicos",
        "Active Weather Warnings",
    )

    def _is_section_line(line: str) -> bool:
        stripped = line.strip().rstrip(":")
        return any(marker.lower() in stripped.lower() for marker in section_markers)

    def _is_day_line(line: str) -> bool:
        stripped = line.strip().rstrip(":")
        if stripped.startswith("📅 "):
            return True
        lowered = stripped.lower()
        return stripped.startswith(day_emojis) and any(token in lowered for token in weekday_tokens)

    def _is_detail_line(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith(detail_prefixes)

    def _is_status_line(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith(("✅", "⚠️", "🟡", "🟠", "🔴", "🌊"))

    def _unwrap_full_line_bold(line: str) -> str:
        stripped = line.strip()
        match = re.match(r"^\*\*(.+)\*\*$", stripped)
        return match.group(1).strip() if match else stripped

    structured_lines: list[str] = []
    inside_day_block = False

    for raw_line in text.splitlines():
        stripped = _unwrap_full_line_bold(raw_line)
        if not stripped:
            continue

        if _SOURCE_LINE_RE.match(stripped):
            if structured_lines and structured_lines[-1] != "":
                structured_lines.append("")
            structured_lines.append(stripped)
            inside_day_block = False
            continue

        if stripped == "---":
            if structured_lines and structured_lines[-1] != "":
                structured_lines.append("")
            structured_lines.extend(["---", ""])
            inside_day_block = False
            continue

        if _is_section_line(stripped):
            if structured_lines and structured_lines[-1] != "":
                structured_lines.append("")
            structured_lines.extend([f"**{stripped.rstrip(':')}**", ""])
            inside_day_block = False
            continue

        if _is_day_line(stripped):
            structured_lines.append(f"- **{stripped.rstrip(':')}**")
            inside_day_block = True
            continue

        if _is_detail_line(stripped):
            prefix = "  - " if inside_day_block else "- "
            structured_lines.append(f"{prefix}{stripped}")
            continue

        if _is_status_line(stripped):
            prefix = "  - " if inside_day_block else "- "
            structured_lines.append(f"{prefix}{stripped}")
            inside_day_block = False
            continue

        structured_lines.append(stripped)
        inside_day_block = False

    structured = clean_newlines("\n".join(structured_lines)).strip()
    structured = re.sub(
        r"(?m)^\*\*([✅⚠️🟡🟠🔴🌊][^*]+)\*\*$",
        r"- \1",
        structured,
    )
    structured = re.sub(
        r"(?m)^\*\*(🌤️\s+(?:As condições meteorológicas são normais|Weather conditions are normal)\.?)\*\*$",
        r"- \1",
        structured,
    )
    return structured.strip()


def canonicalize_transport_terms(text: str, language: str = "en") -> str:
    """Normalizes common transport-summary labels to English when needed."""
    if not text or language != "en":
        return text

    replacements = [
        (r"Situação dos Transportes de Lisboa", "Lisbon Transport Status"),
        (r"\bAtualizado:\b", "Updated:"),
        (r"\bAtualizado às\b", "Updated at"),
        (r"\*\*Estado\*\*:", "**Status**:"),
        (r"\*\*Estado das Linhas:\*\*", "**Line Status:**"),
        (r"\*\*Comboio:", "**Train:"),
        (r"\*\*RESUMO DA VIAGEM\*\*", "**TRIP SUMMARY**"),
        (r"Linha:", "Line:"),
        (r"Dura[cç][aã]o:", "Duration:"),
        (r"\*\*Pr[oó]ximas\s+(\d+)\s+Partidas:\*\*", r"**Next \1 Departures:**"),
        (r"\bOutras linhas\b", "Other lines"),
        (r"Circulação normal em todas as linhas", "Normal service on all lines"),
        (r"\*\*Veículos em serviço\*\*:", "**Vehicles in service**:"),
        (r"\*\*Alertas ativos\*\*:", "**Active alerts**:"),
        (r"\*\*Comboios a circular na AML\*\*:", "**Trains running in AML**:"),
        (r"\*\*Comboios com atrasos > 1 min\*\*:", "**Trains with delays > 1 min**:"),
        (r"\*\*Tempo total estimado:\*\*", "**Estimated total time:**"),
        (r"\*\*O seu Trajeto de Metro:\*\*", "**Your Metro Route:**"),
        (r"\*\*Próximos Metros\*\* \(tempo real\)", "**Next Metros** (real time)"),
        (r"\*\*Próximo Metro em:\*\*", "**Next Metro in:**"),
        (r"\*\*Fonte:\*\*", "**Source:**"),
        (r"\bEmbarque na estação\b", "Board at"),
        (r"\bTransferência em\b", "Transfer at"),
        (r"\bSaia na estação\b", "Exit at"),
        (r"\bSiga a pé para\b", "Walk to"),
        (r"\bDireção\b", "Direction"),
        (r"\bSem dados em tempo real\b", "No real-time data available"),
        (r"Próximas Chegadas", "Next Arrivals"),
        (r"Paragens Carris", "Carris Stops"),
        (r"\bParagem\b", "Stop"),
        (r"\bHora:\b", "Time:"),
        (r"\*\*Hora\*\*:", "**Time**:"),
        (r"\bA mostrar\b", "Showing"),
        (r"Usa o ID da paragem com carris_get_arrivals para ver chegadas em tempo real\.", "Use the stop ID to check real-time arrivals."),
        (r"\bAutocarro\b", "Bus"),
        (r"\bElétrico\b", "Tram"),
        (r"\bEletrico\b", "Tram"),
        (r"\bPróxima paragem:\b", "Next stop:"),
        (r"\*\*Próxima paragem\*\*:", "**Next stop**:"),
        (r"\bMatrícula:\b", "Plate:"),
        (r"\*\*Matrícula\*\*:", "**Plate**:"),
        (r"\bFaltam\s+(\d+)\s+paragens\b", r"\1 stops remaining"),
        (r"\bVeículos? a caminho\b", "vehicles on the way"),
        (r"\bTempo viagem estimado:\b", "Estimated travel time:"),
        (r"\badiantado\s+(\d+)\s+min\b", r"\1 min early"),
        (r"\batrasado \+(\d+)\s+min\b", r"\1 min late"),
        (r"\bDados de:\b", "Feed timestamp:"),
        (r"\bFrequência da Linha\b", "Route Frequency"),
        (r"\bAutocarros\b", "Buses"),
        (r"\bTerminais\b", "Terminals"),
        (r"\*\*Terminais\*\*:", "**Terminals**:"),
        (r"\*\*Como usar:\*\*", "**How to use it:**"),
        (r"Procure pelo n[uú]mero da linha \(ex: \*\*([^*]+)\*\*\) na (?:paragem|Stop)", r"Look for the line number (e.g. **\1**) at the stop"),
        (r"Verifique a (?:dire[cç][aã]o|Direction) do (?:autocarro|Bus)", "Check the bus direction"),
        (r"Hor[aá]rios e paragens", "Schedules and stops"),
        (r"\*\*Hor[aá]rios\*\*:", "**Schedules**:"),
        (r"Bilhetes:", "Tickets:"),
        (r"\*\*(\d+) linha\(s\) direta\(s\) encontrada\(s\):\*\*", r"**\1 direct line(s) found:**"),
        (r"Alguns\s+comboios\s+com\s*\+(\d+)min atraso", r"Some trains are delayed by \1 min"),
        (r"Alguns\s+t*trains?\s+com\s*\+(\d+)min atraso", r"Some trains are delayed by \1 min"),
        (r"ou estação", "or station"),
        (r"Partidas restantes Today", "Remaining departures today"),
        (r"\bHoje\b", "Today"),
        (r"\bParagem:\b", "Stop:"),
        (r"\bTotal de passagens hoje:\b", "Total departures today:"),
        (r"\bpassagem\b", "departure"),
        (r"\bpassagens\b", "departures"),
        (r"\bPara\b", "To"),
        (r"(\*\*\[[^\]]+\]\*\*\s+)Para\b", r"\1To"),
        (r"->\s+([^\n]+?)\s*/\s*circula[cç][aã]o", r"-> \1 / circular service"),
        (r"Restauradoures", "Restauradores"),
        (r":\s*para\s+", ": to "),
        (r"\bveículos\b", "vehicles"),
        (r"\balertas\b", "alerts"),
        (r"\bcomboios\b", "trains"),
        (r"\*\*Fonte:\*\*", "**Source:**"),
    ]

    normalized = text
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


def canonicalize_local_information_terms(text: str, language: str = "en") -> str:
    """Normalizes common PT-PT labels frequently leaked into EN local-information outputs."""
    if not text or language != "en":
        return text

    replacements = [
        (r"\*\*Breve descri(?:ç|c)[aã]o\*\*:", "**Brief description**:"),
        (r"\*\*Morada\*\*:", "**Address**:"),
        (r"\*\*Localiza(?:ç|c)[aã]o\*\*:", "**Location**:"),
        (r"\*\*Hor[aá]rio\*\*:", "**Opening hours**:"),
        (r"\*\*Hor[aá]rios de funcionamento\*\*:", "**Opening hours**:"),
        (r"\*\*Dica r[aá]pida\*\*:", "**Quick tip**:"),
        (r"\*\*Dica\*\*:", "**Tip**:"),
        (r"\*\*Pre(?:ç|c)o\*\*:", "**Price**:"),
        (r"\*\*Pre(?:ç|c)os\*\*:", "**Prices**:"),
        (r"\*\*Comprar bilhetes(?:/mais info)?\*\*:", "**Buy tickets**:"),
        (r"\*\*Site Oficial\*\*", "**Official page**"),
        (r"\bHor[aá]rios de funcionamento:\s*consultar website oficial\.?", "Opening hours: check the official website."),
        (r"\bPre(?:ç|c)os?:\s*verificar no local ou website(?: oficial)?\.?", "Prices: check on site or on the official website."),
        (r"\bverificar no local ou website(?: oficial)?\b", "check on site or on the official website"),
        (r"\bconsultar website oficial\b", "check the official website"),
        (r"\bHoje\b", "Today"),
        (r"\bFechado\b", "Closed"),
        (r"\bN[aã]o especificado\b", "Not specified"),
        (r"\*\*Atualizado\*\*:", "**Updated**:"),
        (r"\*\*Atualizado:\*\*", "**Updated:**"),
        (r"\*\*Fonte\*\*:", "**Source**:"),
        (r"\*\*Fonte:\*\*", "**Source:**"),
    ]

    normalized = text
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


def clean_researcher_tool_artifacts(text: str) -> str:
    """Removes raw tool-only summary blocks and duplicated metadata labels."""
    if not text:
        return text

    artifact_patterns = [
        re.compile(r"^(?:[^A-Za-z0-9#]*\s*)?\*\*Found .*\*\*:?$", re.IGNORECASE),
        re.compile(r"^(?:[^A-Za-z0-9#]*\s*)?\*\*Date range:\*\*.*$", re.IGNORECASE),
        re.compile(r"^(?:[^A-Za-z0-9#]*\s*)?\*\*Today is:\*\*.*$", re.IGNORECASE),
        re.compile(r"^(?:[^A-Za-z0-9#]*\s*)?\*\*(?:Total|Sources):\*\*.*$", re.IGNORECASE),
        re.compile(r"^(?:[^A-Za-z0-9#]*\s*)?\*\*Hybrid search:\*\*.*$", re.IGNORECASE),
        re.compile(r"^(?:[^A-Za-z0-9#]*\s*)?(?:Try more specific queries.*|Showing top .*|Podes perguntar-me.*)$", re.IGNORECASE),
        re.compile(r"^\*\*(?:Name|Url|Category|Short Description|Brief description)\*\*:.*$", re.IGNORECASE),
    ]

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(pattern.match(stripped) for pattern in artifact_patterns):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def strip_unconfirmed_accessibility_claims(text: str, language: str = "en") -> str:
    """Removes unsupported accessibility claims and replaces them with a verification note."""
    if not text:
        return text

    kept_lines = []
    removed_claim = False
    for line in text.splitlines():
        stripped = line.strip()
        if _ACCESSIBILITY_CLAIM_RE.search(stripped):
            if re.search(r"\b(check|verify|confirm|not confirmed|n[aã]o confirmado|n[aã]o confirmadas?)\b", stripped, re.IGNORECASE):
                kept_lines.append(line)
            else:
                removed_claim = True
            continue
        kept_lines.append(line)

    cleaned = "\n".join(kept_lines).strip()
    if not removed_claim:
        return cleaned

    note = (
        "⚠️ Accessibility details are not confirmed in the available data, so please verify them on the official venue or operator page."
        if language == "en"
        else "⚠️ Os detalhes de acessibilidade não estão confirmados nos dados disponíveis, por isso confirme-os na página oficial do local ou operador."
    )

    if note in cleaned:
        return cleaned

    lines = cleaned.splitlines() if cleaned else []
    inserted = False
    result = []
    for line in lines:
        if not inserted and _SOURCE_LINE_RE.match(line.strip()):
            result.append(note)
            result.append("")
            inserted = True
        result.append(line)

    if not inserted:
        if result:
            result.append("")
        result.append(note)

    return "\n".join(result).strip()


def infer_researcher_source_kind(user_query: str = "", text: str = "") -> Optional[str]:
    """
    Infers whether a researcher response is primarily about places or events.

    Args:
        user_query: Original user query.
        text: Current response text.

    Returns:
        Optional[str]: `events`, `places`, or None.
    """
    user_query_lower = (user_query or "").lower()
    text_lower = (text or "").lower()

    if user_query_lower:
        if _EVENT_HINTS_RE.search(user_query_lower):
            return "events"
        if _PLACE_HINTS_RE.search(user_query_lower):
            return "places"

    combined = "\n".join(part for part in [user_query_lower, text_lower] if part)
    if not combined:
        return None

    if "/eventos" in combined or "/events" in combined or _EVENT_HINTS_RE.search(combined):
        return "events"
    if "/locais" in combined or "/places" in combined or _PLACE_HINTS_RE.search(combined):
        return "places"
    return None


def canonicalize_visitlisboa_source_line(
    text: str,
    user_query: str = "",
    language: str = "en",
) -> str:
    """
    Normalizes VisitLisboa source labels so they reflect whether the answer is
    about places or events, in the correct user-facing language.

    Args:
        text: Existing response text.
        user_query: Original user query.
        language: Preferred response language.

    Returns:
        str: Updated response text.
    """
    if not text:
        return text

    lower_text = text.lower()
    kind = infer_researcher_source_kind(user_query=user_query, text=text)
    has_visitlisboa = "visitlisboa" in lower_text
    has_lisboa_aberta = (
        "lisboa aberta" in lower_text
        or "open data:" in lower_text
        or "dados abertos" in lower_text
        or "dados.cm-lisboa.pt" in lower_text
    )
    visitlisboa_source_exists = any(
        _SOURCE_LINE_RE.match(line.strip()) and "visitlisboa" in line.lower()
        for line in text.splitlines()
    )

    if not kind:
        return text

    if not has_visitlisboa and not visitlisboa_source_exists:
        return text

    if kind == "events":
        if language == "pt":
            replacement = "📌 **Fonte:** [*VisitLisboa Eventos*](https://www.visitlisboa.com/pt-pt/eventos)"
        else:
            replacement = "📌 **Source:** [*VisitLisboa Events*](https://www.visitlisboa.com/en/events)"
    else:
        if language == "pt":
            replacement = "📌 **Fonte:** [*VisitLisboa Locais*](https://www.visitlisboa.com/pt-pt/locais)"
        else:
            replacement = "📌 **Source:** [*VisitLisboa Places*](https://www.visitlisboa.com/en/places)"

        if has_lisboa_aberta:
            if language == "pt":
                replacement += " | [*Lisboa Aberta*](https://dados.cm-lisboa.pt/)"
            else:
                replacement += " | [*Lisboa Aberta*](https://dados.cm-lisboa.pt/)"

    return _replace_source_line(
        text,
        replacement,
        predicate=lambda line: bool(_SOURCE_LINE_RE.match(line.strip())) and "visitlisboa" in line.lower(),
    )


def canonicalize_planner_source_line(text: str, language: str = "en") -> str:
    """Normalizes planner source lines into a clean multi-source format."""
    if not text:
        return text

    lower_text = text.lower()
    sources = []
    if "visitlisboa" in lower_text:
        sources.append("[*VisitLisboa*](https://www.visitlisboa.com)")
    if "ipma" in lower_text:
        sources.append("[*IPMA*](https://www.ipma.pt)")
    if "metrolisboa" in lower_text:
        sources.append("[*Metro de Lisboa*](https://www.metrolisboa.pt)")
    if "carris.pt" in lower_text or " carris" in lower_text:
        sources.append("[*Carris*](https://www.carris.pt)")
    if "cp.pt" in lower_text or "comboios" in lower_text:
        sources.append("[*CP*](https://www.cp.pt)")

    if not sources:
        return text

    deduped_sources = []
    for source in sources:
        if source not in deduped_sources:
            deduped_sources.append(source)

    timestamp = extract_update_time(text) or datetime.now().strftime("%H:%M")
    if language == "pt":
        replacement = f"📌 **Fonte:** {' | '.join(deduped_sources)} | **Atualizado:** {timestamp}"
    else:
        replacement = f"📌 **Source:** {' | '.join(deduped_sources)} | **Updated:** {timestamp}"

    return _replace_source_line(text, replacement)


def finalize_worker_response(
    text: str,
    agent_name: str,
    user_query: str = "",
    language: Optional[str] = None,
) -> str:
    """
    Applies deterministic post-processing to direct worker outputs so they are
    as safe and polished as the multi-agent final formatter.

    Args:
        text: Worker response text.
        agent_name: Worker name (`weather`, `researcher`, `planner`, etc.).
        user_query: Original user query.
        language: Optional explicit language code.

    Returns:
        str: Finalized worker response.
    """
    if not text or not isinstance(text, str):
        return text or ""

    preferred_language = language or infer_response_language(
        user_query=user_query,
        context_text=text,
        default="en",
    )

    weather_timestamp = None
    text_for_formatting = text
    if agent_name == "weather":
        weather_timestamp = extract_update_time(text)
        text_for_formatting = "\n".join(
            line for line in text.splitlines()
            if not _SOURCE_LINE_RE.match(line.strip())
        ).strip()

    finalized = strip_unsupported_closing_offers(text_for_formatting)
    finalized = format_response(finalized)

    if agent_name == "weather":
        weather_timestamp = weather_timestamp or extract_update_time(finalized)
        finalized = canonicalize_weather_terms(finalized, language=preferred_language)
        finalized = strip_weather_update_lines(finalized)
        finalized = structure_weather_markdown(finalized)
        finalized = canonicalize_weather_source_line(
            finalized,
            language=preferred_language,
            timestamp=weather_timestamp,
        )
    elif agent_name == "researcher":
        finalized = clean_researcher_tool_artifacts(finalized)
        if _ACCESSIBILITY_QUERY_RE.search(user_query or ""):
            finalized = strip_unconfirmed_accessibility_claims(
                finalized,
                language=preferred_language,
            )
        finalized = canonicalize_local_information_terms(finalized, language=preferred_language)
        finalized = canonicalize_visitlisboa_source_line(
            finalized,
            user_query=user_query,
            language=preferred_language,
        )
    elif agent_name in {"planner", "transport"}:
        finalized = strip_unsupported_closing_offers(finalized)
        finalized = canonicalize_local_information_terms(finalized, language=preferred_language)
        if agent_name == "transport":
            finalized = canonicalize_transport_terms(finalized, language=preferred_language)
        else:
            finalized = canonicalize_planner_source_line(finalized, language=preferred_language)

    return clean_newlines(finalized).strip()


def normalize_source_links(text: str) -> str:
    """
    Normalizes malformed HTML anchor tags and bare Metro de Lisboa source text
    into standard markdown links that Streamlit renders correctly.

    Args:
        text: Raw LLM response text.

    Returns:
        str: Text with standardized Metro de Lisboa source links.
    """
    if not text:
        return text

    # Convert malformed/HTML anchors for Metro de Lisboa to markdown.
    text = re.sub(
        r'<a\s+href="?(?:https?://)?metrolisboa\.pt"?[^>]*>\s*Metro de Lisboa\s*</a>',
        r'[*Metro de Lisboa*](https://www.metrolisboa.pt)',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'<a\s+href="?https?://www\.metrolisboa\.pt"?[^>]*>\s*Metro de Lisboa\s*</a>',
        r'[*Metro de Lisboa*](https://www.metrolisboa.pt)',
        text,
        flags=re.IGNORECASE,
    )
    return text


def normalize_metro_terminology(text: str) -> str:
    """
    Fixes incorrect rail terminology when the response is clearly about
    Metro de Lisboa routes.

    Args:
        text: Raw LLM response text.

    Returns:
        str: Text with metro terminology normalized.
    """
    if not text:
        return text

    metro_context = re.search(
        r'(O seu Trajeto de Metro|Próximos Metros|Metro de Lisboa|Linha Azul|Linha Verde|Linha Amarela|Linha Vermelha)',
        text,
        re.IGNORECASE,
    )
    cp_context = re.search(r'\bCP\b|Comboios de Portugal|CP Trains', text, re.IGNORECASE)

    if metro_context and not cp_context:
        replacements = [
            (r'\bcomboios\b', 'metros'),
            (r'\bComboios\b', 'Metros'),
            (r'\bcomboio\b', 'metro'),
            (r'\bComboio\b', 'Metro'),
            (r'\btrems\b', 'metros'),
            (r'\bTrems\b', 'Metros'),
            (r'\btrem\b', 'metro'),
            (r'\bTrem\b', 'Metro'),
            (r'transferência provável', 'transferência'),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)

    return text


def normalize_headers(text: str) -> str:
    """
    Normalizes markdown headers to consistent levels.

    Rules:
        - # (h1) → ### (h3) to avoid oversized headers in Streamlit
        - ## (h2) → ### (h3) for consistency
        - #### and deeper stay as-is

    Args:
        text: Raw markdown text.

    Returns:
        str: Text with normalized header levels.
    """
    # First: convert setext-style headers (underline with === or ---) to ATX style
    # "Title\n=====" -> "# Title"  and  "Title\n-----" -> "## Title"
    text = re.sub(r'^(.+)\n={3,}\s*$', r'### \1', text, flags=re.MULTILINE)
    text = re.sub(r'^(.+)\n-{3,}\s*$', r'### \1', text, flags=re.MULTILINE)

    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        # Only normalize headers at beginning of line (not in code blocks)
        # Use re.sub to safely replace header prefix without eating content chars
        if stripped.startswith("# ") and not stripped.startswith("## "):
            # h1 -> h3: replace the leading '# ' with '### '
            result.append(re.sub(r'^# ', '### ', stripped))
        elif stripped.startswith("## ") and not stripped.startswith("### "):
            # h2 -> h3: replace the leading '## ' with '### '
            result.append(re.sub(r'^## ', '### ', stripped))
        else:
            result.append(line)
    return "\n".join(result)


def add_section_separators(text: str) -> str:
    """
    Adds horizontal rules between major content sections.

    Inserts --- before ### headers (unless one already exists),
    creating visual separation between sections in Streamlit.

    Args:
        text: Markdown text.

    Returns:
        str: Text with separators added between sections.
    """
    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Add separator before ### headers (not the first one)
        if stripped.startswith("### ") and i > 0:
            # Check if previous non-empty line is already a separator
            prev_non_empty = ""
            for j in range(i - 1, -1, -1):
                if lines[j].strip():
                    prev_non_empty = lines[j].strip()
                    break
            if prev_non_empty != "---":
                result.append("")
                result.append("---")
                result.append("")
        result.append(line)
    return "\n".join(result)


def clean_newlines(text: str) -> str:
    """
    Removes excessive consecutive blank lines (max 1 blank line between content).

    Args:
        text: Text with potentially excessive newlines.

    Returns:
        str: Text with at most 2 consecutive newlines (1 blank line).
    """
    # Replace 3+ consecutive newlines with 2 (single blank line)
    return re.sub(r"\n{3,}", "\n\n", text)


def normalize_bullets(text: str) -> str:
    """
    Normalizes bullet point styles to consistent format, ensures labels are bold,
    and adds tight spacing using markdown hard breaks.

    Rules:
    - Lists with emojis do not get standard bullets, they use the emoji.
    - Numbered lists are bolded automatically (e.g., '1.' -> '**1.**')
    - Labels (e.g., 'Morada:', 'Preço:') are bolded automatically.
    - Two spaces are appended to lists and sub-items for tight <br> spacing.
    - Removes dummy TripAdvisor '⭐ 4.5/5' appended to all events
    - Suppresses repeated '⚠️ Nota:' remarks from IPMA.

    Args:
        text: Text to format.

    Returns:
        str: Formatted text.
    """
    lines = text.split("\n")
    out = []
    
    # Match labels (e.g. "Data/Hora:", "Preço: ") optionally prefixed by emoji
    label_pattern = re.compile(r'^([\u2600-\U0010ffff\u2B50\u200D\uFE0F]{1,3}\s*)?([A-Za-zÀ-ÿ/\s]{3,25}):\s*(.*)')
    
    # Matches the useless ratings added by VisitLisboa tool
    remove_stars_pattern = re.compile(r'\s*-\s*⭐\s*4\.5/5\s*$')
    # Filter repeated 'Nota:' elements
    filter_nota_pattern = re.compile(r'^(?:⚠️\s*)?Nota:', re.IGNORECASE)
    
    nota_count = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
            
        m_nota = filter_nota_pattern.match(stripped)
        if m_nota:
            nota_count += 1
            if nota_count > 1:
                continue
                
        # Remove dummy stars
        stripped = remove_stars_pattern.sub('', stripped)
            
        indent = len(line) - len(line.lstrip())
        spaces = " " * indent
        
        # Determine if it's a bulleted line
        is_bullet = stripped.startswith("- ") or stripped.startswith("* ") or stripped.startswith("• ")
        
        if is_bullet:
            content = stripped[2:].strip()
        else:
            content = stripped
            
        # Detect numbered lists and format labels (Data/Hora: -> **Data/Hora**: )
        if "**" not in content and not content.startswith("#"):
            m_num = re.match(r'^(\d+\.)\s+(.*)', content)
            if m_num:
                num = m_num.group(1)
                rest = m_num.group(2)
                content = f"**{num}** {rest}"
            else:
                m_label = label_pattern.match(content)
                if m_label:
                    emoji_part = m_label.group(1) or ""
                    label = m_label.group(2).strip()
                    rest = m_label.group(3)
                    lowered_label = label.lower()
                    lowered_rest = rest.lower()
                    looks_like_url_prefix = lowered_label in {"http", "https"} or lowered_rest.startswith("//")
                    if not looks_like_url_prefix:
                        content = f"{emoji_part}**{label}**: {rest}"
                
        # Format the output block
        if is_bullet:
            # Normalize all bullet variants to standard Markdown "- "
            out.append(f"{spaces}- {content}")
        else:
            # Non-bullet line: use modified content (with auto-bolded labels/numbers)
            # if content was changed, rebuild the line preserving indentation
            if content != stripped:
                out.append(f"{spaces}{content}")
            else:
                out.append(line)
                
    return "\n".join(out)


def ensure_clickable_urls(text: str) -> str:
    """
    Wraps bare URLs in markdown link syntax for clickable rendering.

    Detects URLs that aren't already in markdown link format [text](url)
    and wraps them. Skips URLs already inside markdown links or code blocks.

    Args:
        text: Text potentially containing bare URLs.

    Returns:
        str: Text with all URLs made clickable.
    """
    # Match bare URLs not preceded by ]( (already a markdown link target)
    url_pattern = re.compile(r'(?<!\]\()(https?://[^\s\)\]]+)')
    # Pattern for existing markdown links: [text](url)
    md_link_pattern = re.compile(r'\[[^\]]*\]\([^)]+\)')

    def replace_url(match):
        url = match.group(1)
        try:
            domain = urlparse(url).netloc
            if domain.startswith("www."):
                domain = domain[4:]
            return f"[{domain}]({url})"
        except Exception:
            return f"[Link]({url})"

    lines = text.split("\n")
    result = []
    in_code_block = False

    for line in lines:
        # Skip code blocks
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            continue

        if in_code_block:
            result.append(line)
            continue

        # Find existing markdown links in the line and protect their URLs
        # Replace bare URLs only outside of markdown link constructs
        existing_links = [(m.start(), m.end()) for m in md_link_pattern.finditer(line)]

        if not existing_links:
            # No existing links - safe to replace all bare URLs
            result.append(url_pattern.sub(replace_url, line))
        else:
            # Has existing links - only replace URLs outside of them
            new_line = []
            last_end = 0
            for start, end in existing_links:
                # Process text before the markdown link (may have bare URLs)
                segment = line[last_end:start]
                new_line.append(url_pattern.sub(replace_url, segment))
                # Keep the markdown link as-is
                new_line.append(line[start:end])
                last_end = end
            # Process remaining text after last markdown link
            segment = line[last_end:]
            new_line.append(url_pattern.sub(replace_url, segment))
            result.append("".join(new_line))

    return "\n".join(result)


def strip_internal_sections(text: str) -> str:
    """
    Removes sections that expose internal system details (QA, disclaimers, etc.)
    that should never appear in user-facing responses.

    Matches header-based sections like:
        - ### Observações e disclaimers
        - ### Checklist de Completude
        - ### Quality Check
        - ### QA Results / Data Validation
        - ### Fonte & Observações (when it contains QA content)

    Args:
        text: Formatted markdown text.

    Returns:
        str: Text with internal sections removed.
    """
    # Patterns for internal section headers (case-insensitive)
    internal_patterns = [
        r'observa[çc][õo]es\s+e\s+disclaimers',
        r'checklist\s+de\s+completude',
        r'quality\s+(check|assurance)',
        r'qa[\s_]+(results?|validation|disclaimers?)',
        r'data\s+validation',
        r'completeness\s+check',
        r'disclaimers?\s*$',
        r'notas?\s+de\s+qualidade',
        r'controlo\s+de\s+qualidade',
    ]
    combined = '|'.join(internal_patterns)
    # Match headers (any level) containing these patterns
    header_re = re.compile(
        r'^(#{1,6})\s+.*?(' + combined + r').*$',
        re.IGNORECASE | re.MULTILINE
    )

    lines = text.split('\n')
    result = []
    skip_until_next_header = False
    skip_header_level = 0

    for line in lines:
        stripped = line.strip()

        # Check if this is a header
        header_match = re.match(r'^(#{1,6})\s+', stripped)

        if header_match:
            level = len(header_match.group(1))

            if skip_until_next_header:
                # Found a new header - check if same or higher level (stop skipping)
                if level <= skip_header_level:
                    skip_until_next_header = False
                else:
                    # Sub-header of the skipped section, continue skipping
                    continue

            # Check if this header matches an internal pattern
            if header_re.match(stripped):
                skip_until_next_header = True
                skip_header_level = level
                continue

        if skip_until_next_header:
            continue

        result.append(line)

    return '\n'.join(result)


def add_section_spacing(text: str) -> str:
    """
    Ensures blank lines before section-like markers so content blocks
    are visually separated (e.g. Avisos, Dicas, Fonte, Nota, headers).

    This prevents different content blocks from appearing 'cramped'
    together without any breathing room.

    Args:
        text: Markdown text to process.

    Returns:
        str: Text with proper spacing before section markers.
    """
    # Patterns that should always have a blank line before them
    section_markers = [
        r"^#{1,4}\s",                     # Any markdown header
        r"^(?:⚠️|⚠)\s*\*\*(?:Avisos|Aviso|Warnings?|Nota|Note)",
        r"^💡\s*\*\*(?:Dicas?|Tips?|Sugest)",
        r"^📌\s*\*\*(?:Fonte|Source)",
        r"^🌡️",                           # Weather emoji section
        r"^🌤️",
        r"^🌧️",
        r"^---\s*$",                       # Horizontal rules
    ]
    combined = "|".join(f"(?:{p})" for p in section_markers)
    section_re = re.compile(combined)

    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i > 0 and section_re.match(stripped):
            # Check if previous line is already blank
            prev = result[-1].strip() if result else ""
            if prev != "":
                result.append("")
        result.append(line)

    return "\n".join(result)


def clean_decorative_separators(text: str) -> str:
    """
    Removes decorative separator lines that are not valid markdown.

    Lines consisting only of repeated '=' or '-' characters (5 or more)
    are removed. Valid markdown horizontal rules ('---' exactly 3 dashes
    on a line) are preserved.

    This acts as a safety net for tool outputs that use decorative
    separators like '=' * 50 or '-' * 30 which render as plain text
    in Streamlit rather than as visual dividers.

    Args:
        text: Text potentially containing decorative separators.

    Returns:
        str: Text with decorative separators removed.
    """
    # Remove lines of 5+ repeated = or - characters (but preserve --- which is valid markdown)
    # Also preserve lines with mixed content (e.g., '--- some text')
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are ONLY repeated = or - (5+ chars), these are decorative
        if len(stripped) >= 5 and all(c in '=-' for c in stripped):
            continue
        result.append(line)
    return "\n".join(result)


def generate_response_title(
    agents_called: list,
    user_query: str,
    language: str = "en",
) -> Optional[str]:
    """
    Generates a contextual ### (h3) title for the response based on routing.

    Returns None for direct responses (no agents), planner responses,
    or greetings, so they remain untitled.

    Args:
        agents_called: List of agent names invoked (e.g., ["weather"]).
        user_query: The original user query.
        language: Language code ('en' or 'pt').

    Returns:
        Optional[str]: A markdown ### title string, or None.
    """
    if not agents_called:
        return None  # Direct response (OOS, greeting) - no title

    if "planner" in agents_called:
        return None  # Planner generates its own header

    query_lower = user_query.lower()

    # --- Single agent responses ---
    if len(agents_called) == 1:
        agent = agents_called[0]

        if agent == "weather":
            return (
                "### \U0001f324\ufe0f Previsão Meteorológica"
                if language == "pt"
                else "### \U0001f324\ufe0f Weather Forecast"
            )

        elif agent == "transport":
            return (
                "### \U0001f687 Informação de Transportes"
                if language == "pt"
                else "### \U0001f687 Transport Information"
            )

        elif agent == "researcher":
            # Keyword-based subcategorization
            event_kw = [
                "evento", "event", "concerto", "concert", "festival",
                "espetáculo", "show", "teatro", "theatre", "theater",
                "ópera", "opera", "dança", "dance", "exposição", "exhibition",
            ]
            place_kw = [
                "museu", "museum", "monumento", "monument", "castelo", "castle",
                "igreja", "church", "torre", "tower", "praça", "square",
                "bairro", "neighborhood", "miradouro", "viewpoint", "jardim",
                "garden", "parque", "park",
            ]
            food_kw = [
                "restaurante", "restaurant", "comida", "food", "comer", "eat",
                "café", "coffee", "bar", "pastelaria", "bakery",
                "gastronomia", "gastronomy", "nightlife", "vida noturna",
            ]
            service_kw = [
                "farmácia", "pharmacy", "hospital", "escola", "school",
                "biblioteca", "library", "polícia", "police", "bombeiros",
                "fire", "wc", "sanitário", "mercado", "market", "creche",
                "estacionamento", "parking", "feira", "marketplace",
            ]
            history_kw = [
                "história", "history", "cultura", "culture", "origem", "origin",
                "fundação", "founded", "tradição", "tradition",
            ]

            if any(kw in query_lower for kw in event_kw):
                return (
                    "### \U0001f3ad Eventos Culturais"
                    if language == "pt"
                    else "### \U0001f3ad Cultural Events"
                )
            elif any(kw in query_lower for kw in place_kw):
                return (
                    "### \U0001f4cd Locais e Atrações"
                    if language == "pt"
                    else "### \U0001f4cd Places & Attractions"
                )
            elif any(kw in query_lower for kw in food_kw):
                return (
                    "### \U0001f37d\ufe0f Gastronomia"
                    if language == "pt"
                    else "### \U0001f37d\ufe0f Food & Dining"
                )
            elif any(kw in query_lower for kw in service_kw):
                return (
                    "### \U0001f3e5 Serviços Essenciais"
                    if language == "pt"
                    else "### \U0001f3e5 Essential Services"
                )
            elif any(kw in query_lower for kw in history_kw):
                return (
                    "### \U0001f4da História e Cultura"
                    if language == "pt"
                    else "### \U0001f4da History & Culture"
                )
            else:
                return (
                    "### \U0001f4cb Informação Local"
                    if language == "pt"
                    else "### \U0001f4cb Local Information"
                )

    # --- Multi-agent (without planner) - combined titles ---
    if "weather" in agents_called and "transport" in agents_called:
        return (
            "### \U0001f324\ufe0f\U0001f687 Meteorologia e Transportes"
            if language == "pt"
            else "### \U0001f324\ufe0f\U0001f687 Weather & Transport"
        )
    elif "weather" in agents_called:
        return (
            "### \U0001f324\ufe0f Previsão Meteorológica"
            if language == "pt"
            else "### \U0001f324\ufe0f Weather Forecast"
        )
    elif "transport" in agents_called:
        return (
            "### \U0001f687 Informação de Transportes"
            if language == "pt"
            else "### \U0001f687 Transport Information"
        )
    else:
        return (
            "### \U0001f4cb Informação Local"
            if language == "pt"
            else "### \U0001f4cb Local Information"
        )


def ensure_response_title(text: str, title: Optional[str]) -> str:
    """
    Prepends a contextual title to the response if it doesn't already have one.

    Skips prepending if:
        - title is None or empty
        - response already starts with a markdown header (###, ##, #)
        - response already starts with a bold title (**Title**)

    Args:
        text: Formatted response text.
        title: The ### title to prepend, or None.

    Returns:
        str: Response with title prepended (or unchanged).
    """
    if not title or not text:
        return text or ""

    # Check if response already starts with a header or bold title
    first_line = text.strip().split("\n")[0].strip()
    if first_line.startswith("### ") or first_line.startswith("## ") or first_line.startswith("# "):
        return text  # Already has a header
    if re.match(r"^\*\*[^*]+\*\*\s*$", first_line):
        return text  # Already has a bold title line
    if re.match(r"^(?:[🚇🚌🚆🚋🌤️🗺️📚🎭📍]\s+)?\*\*[^*]+\*\*(?:\s*(?::|→|-).*)?$", first_line):
        return text  # Already has a strong emoji/bold title line
    if re.match(r"^[🚇🚌🚆🚋🌤️🗺️📚🎭📍]\s+.+$", first_line):
        return text  # Already starts with an emoji title

    return f"{title}\n\n{text}"



def strip_hallucinations(text: str) -> str:
    if not text:
        return ""

    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        if re.match(r"^(?:\s*|-\s*|\*\s*|\**|\[|\]|\*|#|>)*\s*(Introdu[cç][aã]o|Introduction)\b", line, re.IGNORECASE):
            continue
        if re.match(r"^(?:\s*|-\s*|\*\s*|\**|\[|\]|\*|#|>)*\s*(Contrainte do utilizador|Restri[cç][õo]es do utilizador|How the response meets|Acessibilidade/Tempo/Budget|Accessibility/Time/Budget)\b", line, re.IGNORECASE):
            continue
        if re.match(r"^(?:\s*|-\s*|\*\s*|\**|\[|\]|\*|#|>|⚠️\s*)*\s*(?:\*\*\s*)?(Observa[cç][aã]o|Observa[cç][õo]es|Observation|Nota|Note|Notes?)(?:\s*\*\*)?:?", line, re.IGNORECASE):
            continue
        if re.match(r"^(?:\s*|-\s*|\*\s*|\**|\[|\]|\*|#|>|⚠️\s*)*\s*(?:\*\*\s*)?(Diga se|Se quiser|Se quiseres|Se preferir|Quer que eu|Posso ajudar|Posso detalhar|Posso filtrar|Posso trazer|Posso verificar|I can also|I can help|I can filter|I can fetch|I can bring|If you want, I can|If you['’]d like|Would you like me to|Let me know):?", line, re.IGNORECASE):
            continue
        if re.match(r"^\s*\*\*Source\*\*:\s*VisitLisboa\s+(Places|Events)\s*$", line, re.IGNORECASE):
            continue
        if re.match(r"^\s*\*\*Fonte\*\*:\s*VisitLisboa\s+(Locais|Eventos)\s*$", line, re.IGNORECASE):
            continue
        if re.match(r"^\s*🗓️\s*\[.*weather note.*\]\s*$", line, re.IGNORECASE):
            continue
        if re.match(r"^(?:\s*|-\s*|\*\s*|\**|\[|\]|\*|#|>)*\s*(⭐\s*Rating:\s*(Sem avaliação de rating|No rating available))\s*$", line, re.IGNORECASE):
            continue
        if "Não listado o Opposto" in line or "opposite direction" in line.lower():
            continue
        clean_lines.append(line)
    text = "\n".join(clean_lines)

    # Normalize source emphasis before truncating.
    text = re.sub(r"Fonte:\s*📌\s*Fonte:\s*", "📌 **Fonte:** ", text, flags=re.IGNORECASE)
    text = re.sub(r"^Fonte:\s*", "📌 **Fonte:** ", text, flags=re.MULTILINE)
    text = re.sub(r"📌\s*Fonte:", "📌 **Fonte:**", text)
    text = re.sub(r"\|\s*Atualizado:", "| **Atualizado:**", text)
    text = re.sub(r"\|\s*Updated:", "| **Updated:**", text)
    text = re.sub(r"\*\*\|\s*\*\*(Atualizado|Updated):\*+", r"| **\1:**", text)
    text = text.replace("**| **Atualizado:****", "| **Atualizado:**")
    text = text.replace("**| **Updated:****", "| **Updated:**")

    # Hard truncate after the first valid source line.
    match = re.search(r"^(📌\s*\*\*Fonte:\*\*.*?(?:Atualizado|Updated):\s*\d{2}:\d{2}).*$", text, re.MULTILINE)
    if match:
        text = (text[:match.start()] + match.group(1)).rstrip()
    else:
        match2 = re.search(r"^(📌\s*Fonte:.*?(?:Atualizado|Updated):\s*\d{2}:\d{2}).*$", text, re.MULTILINE)
        if match2:
            text = (text[:match2.start()] + match2.group(1)).rstrip()
        else:
            match3 = re.search(r"^(📌\s*\*\*Fonte:\*\*.*)$", text, re.MULTILINE)
            if match3:
                text = (text[:match3.start()] + match3.group(1)).rstrip()

    return text


def format_response(text: str) -> str:
    """
    Main formatting pipeline for LLM responses.

    Applies all formatting transformations in order:
        1. Strip internal/QA sections that should never reach the user
        2. Clean decorative separators (e.g., '=' * 50, '-' * 30)
        3. Normalize headers (avoid h1/h2, use h3+)
        4. Add spacing between distinct content sections
        5. Add section separators (---) before ### headers
        6. Clean excessive newlines (after all spacing steps)
        7. Normalize bullet styles
        8. Ensure URLs are clickable

    Args:
        text: Raw LLM response text.

    Returns:
        str: Formatted text ready for Streamlit rendering.
    """
    if not text or not isinstance(text, str):
        return text or ""

    text = normalize_source_links(text)
    text = normalize_metro_terminology(text)
    text = strip_hallucinations(text)
    text = strip_internal_sections(text)
    text = clean_decorative_separators(text)
    text = normalize_headers(text)
    text = add_section_spacing(text)
    text = add_section_separators(text)
    text = clean_newlines(text)
    text = normalize_bullets(text)
    text = ensure_clickable_urls(text)

    return text.strip()


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    import time

    test_input = """# Weather in Lisbon

## Current Conditions

* 🌡️ Temperature: **22°C**
* 💧 Humidity: 65%
• 🌬️ Wind: 15 km/h NW
* Normal bullet without emoji

## What to do today

Here are some suggestions:

* Visit the Jerónimos Monastery
• Take the 28E tram
* Walk along the riverfront

Check the official site: https://www.visitlisboa.com

## Transport Tips

More info at https://www.metrolisboa.pt and https://www.carris.pt

### Already a h3

This should stay as-is.




Too many blank lines above should be reduced.
"""

    print("=" * 60)
    print("🧪 Response Formatter Test")
    print("=" * 60)

    start = time.time()
    output = format_response(test_input)
    elapsed = time.time() - start

    print(f"\n📥 INPUT ({len(test_input)} chars):")
    print("-" * 40)
    print(test_input[:200] + "...")

    print(f"\n📤 OUTPUT ({len(output)} chars, {elapsed*1000:.1f}ms):")
    print("-" * 40)
    print(output)

    # Verify transformations
    checks = {
        "No h1/h2 headers": not any(
            (line.startswith("# ") and not line.startswith("## ")) or
            (line.startswith("## ") and not line.startswith("### "))
            for line in output.split("\n")
        ),
        "No excessive newlines": "\n\n\n" not in output,
        "URLs are clickable": "](http" in output,
    }

    print("\n✅ Checks:")
    all_pass = True
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check}")
        if not result:
            all_pass = False

    # --- generate_response_title() tests ---
    print("\n\033[1m🔤 generate_response_title() Tests:\033[0m")
    # Signature: (agents_called: list, user_query: str, language: str) -> Optional[str]
    title_cases = [
        (["weather"], "weather forecast lisbon", "en", "### "),
        (["weather"], "tempo em lisboa amanhã", "pt", "### "),
        (["transport"], "próximo metro rossio", "pt", "### "),
        (["transport"], "bus schedule to Cascais", "en", "### "),
        (["researcher"], "exposição no museu", "pt", "### "),
        (["researcher"], "museum near alfama", "en", "### "),
        (["researcher"], "jantar no bairro alto", "pt", "### "),
        (["researcher"], "restaurant recommendations", "en", "### "),
        (["planner"], "plan my full day in lisbon", "en", None),
        ([], "olá bom dia", "pt", None),
    ]
    title_pass = 0
    for agents, query, lang, expected in title_cases:
        title = generate_response_title(agents, query, language=lang)
        if expected is None:
            ok = title is None
        else:
            ok = title is not None and title.startswith(expected)
        status = "✅" if ok else "❌"
        print(f"  {status} [{lang}] agents={agents} '{query}' → {title!r}")
        if ok:
            title_pass += 1
        else:
            all_pass = False
    print(f"  → {title_pass}/{len(title_cases)} title tests passed")

    # --- ensure_response_title() tests ---
    print("\n\033[1m📌 ensure_response_title() Tests:\033[0m")
    # Signature: (text: str, title: Optional[str]) -> str
    ensure_cases = [
        ("Some content without a header.", "### 🌤️ Weather in Lisbon", True),
        ("### Existing Header\nContent", "### 🚇 Transport", False),
        ("**Bold Title**\nContent", "### 🎭 Events", False),
        ("Some content", None, False),
        ("", "### 🎭 Events", False),
    ]
    ensure_pass = 0
    for text_in, title_in, expect_injected in ensure_cases:
        result = ensure_response_title(text_in, title_in)
        if expect_injected:
            ok = result.lstrip().startswith("### ") and str(title_in) in result
        elif text_in == "":
            ok = result == ""
        elif title_in is None:
            ok = result == text_in
        else:
            ok = result == text_in
        status = "✅" if ok else "❌"
        label = "(injected)" if expect_injected else "(unchanged)"
        print(f"  {status} {label}: title={str(title_in)[:25]!r} → {result[:50]!r}...")
        if ok:
            ensure_pass += 1
        else:
            all_pass = False
    print(f"  → {ensure_pass}/{len(ensure_cases)} ensure tests passed")

    if all_pass:
        print("\n\033[1;32m🎉 ALL CHECKS PASSED\033[0m")
    else:
        print("\n\033[1;31m❌ SOME CHECKS FAILED\033[0m")
