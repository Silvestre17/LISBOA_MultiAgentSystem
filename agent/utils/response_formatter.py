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
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse

_PT_CATEGORY_VALUE_MAP = {
    "music": "Música",
    "monuments": "Monumentos",
    "museum": "Museu",
    "museums": "Museus",
    "museum & monument": "Museus e Monumentos",
    "museums & monuments": "Museus e Monumentos",
    "view point": "Miradouro",
    "view points": "Miradouros",
    "viewpoint": "Miradouro",
    "viewpoints": "Miradouros",
    "tours": "Visitas guiadas",
    "tour": "Visita guiada",
    "family & kids": "Família e Crianças",
    "family and kids": "Família e Crianças",
    "gardens & parks": "Jardins e Parques",
    "gardens and parks": "Jardins e Parques",
    "nightlife": "Vida noturna",
    "restaurants": "Restaurantes",
    "restaurant": "Restaurante",
    "architecture": "Arquitetura",
    "art": "Arte",
    "history": "História",
    "culture": "Cultura",
    "shopping": "Compras",
}

_PT_DURATION_VALUE_MAP = {
    "single day": "Um só dia",
    "one day": "Um só dia",
    "multiple days": "Vários dias",
    "multi-day": "Vários dias",
    "ongoing": "A decorrer",
    "long term": "Longa duração",
    "temporary": "Temporário",
    "permanent": "Permanente",
}

_SOURCE_LINE_RE = re.compile(r'^(?:[-*•]\s*)?(?:📌\s*)?(?:\*\*)?(?:Fonte|Source)(?:\*\*)?:.*$', re.IGNORECASE)
_PT_LANGUAGE_HINTS_RE = re.compile(
    r"\b(olá|ola|bom dia|boa tarde|boa noite|como|qual|quais|quero|preciso|planeia|planejar|plano|roteiro|sugere|visitar|passeio|museu|museus|evento|eventos|hoje|amanhã|amanha|previsão|tempo|locais|morada|fonte|autocarro|autocarros|comboio|comboios|transportes?|situa[cç][aã]o|d[aá]-?me|bairro|perto)\b",
    re.IGNORECASE,
)
_EN_LANGUAGE_HINTS_RE = re.compile(
    r"\b(hello|hi|good morning|good afternoon|good evening|what|where|when|which|who|why|how|tell me|plan|afternoon|evening|night|trip|visit|around|can you|could you|would you|i want|i need|please|today|tomorrow|weather|forecast|museum|museums|event|events|book fair|train|bus|metro|source|address)\b",
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
_TRANSPORT_WEATHER_BLOCK_RE = re.compile(
    r"\n?[⛈️🌤️☔]\s*\*\*(?:Tempo em Lisboa|Weather in Lisbon|Weather)\*\*\s*\n(?:\s*[-*•].*\n?){1,4}(?=(?:\s*(?:🚇|🚌|🚆|\*\*Opção|\*\*Option|📌|$)))",
    re.IGNORECASE,
)
_TIMED_SECTION_HEADER_RE = re.compile(
    r"^(?:[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s*)?\d{1,2}:\d{2}\s*·\s*.+$"
)
_TRANSPORT_ROUTE_TITLE_RE = re.compile(
    r"^(?:[🚇🚌🚆🚋]\s+)?\*\*[^*]+(?:→|·)[^*]+\*\*(?:\s*(?::|—|-).*)?$"
)
_DISPLAY_TITLE_SMALL_WORDS = {
    "pt": {
        "a", "à", "ao", "aos", "às", "com", "da", "das", "de", "do", "dos", "e",
        "em", "na", "nas", "no", "nos", "o", "os", "ou", "para", "por", "sem",
        "um", "uma", "uns", "umas",
    },
    "en": {
        "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or",
        "the", "to", "via", "with",
    },
}


def _title_case_segment(text: str, language: str) -> str:
    """Apply display-oriented title casing to one compact text segment."""
    if not text:
        return text

    if text.isupper() or re.search(r"\d", text):
        return text

    for separator in ("-", "/"):
        if separator in text:
            return separator.join(_title_case_segment(part, language) for part in text.split(separator))

    lowered = text.lower()
    return lowered[:1].upper() + lowered[1:]


def to_display_title_case(text: str, language: str = "en") -> str:
    """Format headings in a consistent PT/EN display title case."""
    if not text:
        return text

    language = language if language in _DISPLAY_TITLE_SMALL_WORDS else "en"
    small_words = _DISPLAY_TITLE_SMALL_WORDS[language]
    parts = re.split(r"(\s+)", text.strip())
    result: list[str] = []
    word_index = 0

    for part in parts:
        if not part or part.isspace():
            result.append(part)
            continue

        match = re.match(r"^(?P<prefix>[^\wÀ-ÿ]*)(?P<core>[\wÀ-ÿ'.’-]+)(?P<suffix>[^\wÀ-ÿ]*)$", part)
        if not match:
            result.append(part)
            continue

        prefix = match.group("prefix")
        core = match.group("core")
        suffix = match.group("suffix")
        lowered = core.lower()

        if word_index > 0 and lowered in small_words:
            transformed = lowered
        else:
            transformed = _title_case_segment(core, language)

        result.append(f"{prefix}{transformed}{suffix}")
        word_index += 1

    return "".join(result)


def infer_response_language(
    user_query: str = "",
    context_text: str = "",
    default: str = "en",
) -> str:
    """
    Infers the preferred response language from the user query first and the
    existing text second.

    The detector first consults :mod:`langdetect` when available (a small,
    offline ISO-639-1 classifier). Portuguese (PT/BR) maps to ``"pt"``;
    English to ``"en"``; any other detected language maps to ``"en"`` so the
    assistant can serve a universal fallback response (the higher-level
    ``resolve_output_language`` helper also exposes a flag indicating that a
    bilingual note should be surfaced). When ``langdetect`` is unavailable or
    inconclusive, the function falls back to the legacy PT/EN keyword and
    diacritic heuristic so the assistant keeps working without the extra
    dependency.

    Args:
        user_query: Original user query, if available.
        context_text: Response text or context hints.
        default: Fallback language code used only when both langdetect and the
            hint heuristic are inconclusive.

    Returns:
        str: ``"pt"`` or ``"en"``.
    """
    normalized_default = default if default in {"pt", "en"} else "en"

    # langdetect on tiny inputs ("ok", "ok\nok") is noisy, so require at least
    # 15 non-whitespace characters before trusting its verdict.
    _LANG_DETECT_MIN_LEN = 15

    # Portuguese without diacritics is routinely misclassified by langdetect
    # as Spanish/Galician/Catalan because the Romance cognates overlap.
    # Conversely, English responses that embed one Portuguese station or
    # neighborhood name ("estação de Benfica") can flip langdetect to PT.
    # To keep QA and worker language checks stable we trust langdetect only
    # when it returns PT or EN *and* the keyword-based hints are ambiguous.
    def _trusted_iso(raw_text: str) -> Optional[str]:
        core = re.sub(r"\s+", "", raw_text)
        if len(core) < _LANG_DETECT_MIN_LEN:
            return None
        iso = _detect_language_iso(raw_text)
        if iso in {"pt", "pt-br", "pt-pt"}:
            return "pt"
        if iso == "en":
            return "en"
        return None

    def _classify(text: str) -> Optional[str]:
        pt_match = bool(_PT_LANGUAGE_HINTS_RE.search(text))
        en_match = bool(_EN_LANGUAGE_HINTS_RE.search(text))
        has_pt_diacritics = bool(re.search(r"[ãõáàâéêíóôúç]", text, re.IGNORECASE))

        # Strong unilateral keyword signal wins over langdetect.
        if pt_match and not en_match:
            return "pt"
        if en_match and not pt_match:
            return "en"

        iso = _trusted_iso(text)
        if iso:
            return iso

        if pt_match and en_match:
            return "pt"
        if has_pt_diacritics:
            return "pt"
        return None

    if user_query:
        verdict = _classify(user_query)
        if verdict:
            return verdict
        return normalized_default

    combined = context_text.strip()
    if not combined:
        return normalized_default

    verdict = _classify(combined)
    if verdict:
        return verdict
    return normalized_default


# --------------------------------------------------------------------------
# Robust language detection and output-language resolution
# --------------------------------------------------------------------------
# Map ISO-639-1 codes to human-friendly language names used in the bilingual
# note that we surface when the user writes in a language other than PT or EN.
_LANGUAGE_DISPLAY_NAMES = {
    "pt": "Portuguese",
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "nl": "Dutch",
    "ca": "Catalan",
    "gl": "Galician",
    "ro": "Romanian",
    "ru": "Russian",
    "uk": "Ukrainian",
    "pl": "Polish",
    "tr": "Turkish",
    "ar": "Arabic",
    "he": "Hebrew",
    "fa": "Persian",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "el": "Greek",
    "cs": "Czech",
    "sk": "Slovak",
    "hu": "Hungarian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sr": "Serbian",
    "sl": "Slovenian",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "tl": "Filipino",
}


def _detect_language_iso(text: str) -> Optional[str]:
    """Attempt to detect the ISO-639-1 language code for the given text.

    Uses the optional ``langdetect`` dependency when available. The detector
    factory is seeded for deterministic results so evaluation runs stay
    reproducible. Returns ``None`` when the dependency is missing or when the
    library cannot produce a confident guess (e.g., for very short inputs).
    """
    stripped = (text or "").strip()
    if len(stripped) < 3:
        return None
    try:
        from langdetect import DetectorFactory, detect  # type: ignore
        from langdetect.lang_detect_exception import LangDetectException  # type: ignore
    except Exception:
        return None

    try:
        DetectorFactory.seed = 42
        code = detect(stripped)
    except LangDetectException:
        return None
    except Exception:
        return None

    if not code:
        return None
    return str(code).lower().strip()


def _detect_non_latin_script_iso(text: str) -> Optional[str]:
    """Return a coarse ISO code from script ranges without calling langdetect.

    This is intentionally conservative and exists to avoid unstable language
    guesses for non-Latin scripts, where ``langdetect`` can misclassify short
    Chinese text as Korean or similar nearby languages.
    """
    if not text:
        return None
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[\u3400-\u9fff]", text):
        return "zh-cn"
    if re.search(r"[\u0400-\u04ff]", text):
        return "ru"
    if re.search(r"[\u0600-\u06ff]", text):
        return "ar"
    if re.search(r"[\u0590-\u05ff]", text):
        return "he"
    if re.search(r"[\u0e00-\u0e7f]", text):
        return "th"
    if re.search(r"[\u0370-\u03ff]", text):
        return "el"
    return None


def resolve_output_language(
    user_query: str = "",
    ui_default: str = "en",
) -> tuple[str, bool, Optional[str]]:
    """Resolve the final output language for the assistant response.

    The LISBOA assistant is optimized for **Portuguese (PT-PT)** and
    **English**. When the user writes in any other language (French, German,
    Chinese, Japanese, etc.), the system answers in English and surfaces a
    small bilingual note so the user knows why the output language differs
    from the input.

    The decision flow is:
    1. If the query contains explicit PT or EN hints (keywords, diacritics),
       trust the heuristic. This avoids false-positive notes on very short
       Portuguese greetings such as "Ola" that langdetect can misclassify.
    2. Otherwise, ask ``langdetect``. If it reports a non-PT/EN language
       with enough input to be reliable, answer in English with the note.
    3. Fall back to the legacy hint-based heuristic.

    Args:
        user_query: The raw user message.
        ui_default: UI-selected default language ("pt" or "en").

    Returns:
        tuple[str, bool, Optional[str]]: ``(output_language, requires_note,
        detected_iso_or_name)``.
        * ``output_language`` is either ``"pt"`` or ``"en"``.
        * ``requires_note`` is ``True`` when the detected input language is
          neither PT nor EN, signalling that the final response should be
          prepended with the bilingual note.
        * The third element is the detected ISO code (or ``None`` when
          detection fell back to the hint-based heuristic).
    """
    ui_default_norm = ui_default if ui_default in {"pt", "en"} else "en"
    query = (user_query or "").strip()

    # Any non-Latin script (CJK, Cyrillic, Arabic, Hebrew, Greek, Thai, etc.)
    # is an unambiguous signal that the user did not write in PT or EN.
    script_iso = _detect_non_latin_script_iso(query)
    if script_iso:
        iso = script_iso
        return "en", True, iso

    # Spanish-specific punctuation is a strong marker that also disambiguates
    # diacritic-heavy queries that the PT/EN heuristic would otherwise confuse.
    if re.search(r"[¿¡]", query):
        return "en", True, "es"

    # Explicit PT/EN hints take priority so short greetings ("Olá", "Hello")
    # are never flagged as French/Turkish/etc. by langdetect noise.
    pt_hint = bool(_PT_LANGUAGE_HINTS_RE.search(query))
    en_hint = bool(_EN_LANGUAGE_HINTS_RE.search(query))
    # PT-unique diacritics (tilde, cedilla, circumflex) reliably mark PT-PT.
    # Shared Romance accents (á, à, é, í, ó, ú) are NOT sufficient because
    # French and Spanish share them, so we verify those with langdetect below.
    has_pt_unique = bool(re.search(r"[ãõêôç]", query, re.IGNORECASE))
    has_pt_diacritics = bool(re.search(r"[ãõáàâéêíóôúç]", query, re.IGNORECASE))

    if pt_hint and not en_hint:
        return "pt", False, "pt"
    if en_hint and not pt_hint:
        return "en", False, "en"
    if has_pt_unique and not en_hint:
        return "pt", False, "pt"

    # langdetect needs a minimum amount of signal to be reliable.
    if len(query) >= 15:
        iso = _detect_language_iso(query)
        if iso in {"pt", "pt-br", "pt-pt"}:
            return "pt", False, iso
        if iso == "en":
            return "en", False, iso
        if iso and iso not in {"und", "unknown"}:
            return "en", True, iso

    # Only fall back to "shared Romance accents imply PT" when langdetect
    # could not classify the query (too short or ambiguous).
    if has_pt_diacritics and not en_hint:
        return "pt", False, "pt"

    # Fall back to legacy hint heuristic for anything we cannot classify.
    hint_language = infer_response_language(
        user_query=query,
        default=ui_default_norm,
    )
    return hint_language, False, None


def language_display_name(language_code: Optional[str]) -> str:
    """Return a human-friendly display name for an ISO language code."""
    if not language_code:
        return "another language"
    key = str(language_code).lower().strip()
    if key in _LANGUAGE_DISPLAY_NAMES:
        return _LANGUAGE_DISPLAY_NAMES[key]
    # Fall back to the base code before any regional suffix (e.g. "zh-hk").
    base = key.split("-", 1)[0]
    return _LANGUAGE_DISPLAY_NAMES.get(base, "another language")


def build_bilingual_note(detected_language: Optional[str]) -> str:
    """Build the visually styled bilingual note prepended to EN fallback answers.

    Args:
        detected_language: ISO code or free-form name of the detected input
            language, if known.

    Returns:
        str: Markdown quote block. Safe to render directly in Streamlit.
    """
    display = language_display_name(detected_language)
    return (
        "> ℹ️ **This assistant speaks Portuguese and English.**\n"
        f"> Your message was detected as **{display}** — answering in English below.\n"
        "> *Português · English · Type in either language anytime.*"
    )


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
            (r"Active Weather Warnings \(LSB\)", "Active Weather Warnings for Lisbon"),
            (r"Active Weather Warnings \([A-Z]{3}\)", "Active Weather Warnings"),
            (r"\bSem avisos meteorológicos ativos para Lisboa\.\b", "No active weather warnings for Lisbon."),
            (r"\bSem avisos meteorológicos ativos para a área 'LSB'\.\b", "No active weather warnings for Lisbon."),
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
            (r"Active Weather Warnings for Lisbon", "Avisos Meteorológicos para Lisboa"),
            (r"Active Weather Warnings \(LSB\)", "Avisos Meteorológicos para Lisboa"),
            (r"Active Weather Warnings \([A-Z]{3}\)", "Avisos Meteorológicos"),
            (r"\bNo active weather warnings for Lisbon\.", "Sem avisos meteorológicos ativos para Lisboa."),
            (r"\bNo active weather warnings for area 'LSB'\.", "Sem avisos meteorológicos ativos para Lisboa."),
            (r"\bNo active weather warnings for area '[A-Z]{3}'\.", "Sem avisos meteorológicos ativos."),
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
            (r"\bJan\b", "Jan"),
            (r"\bFeb\b", "Fev"),
            (r"\bMar\b", "Mar"),
            (r"\bApr\b", "Abr"),
            (r"\bMay\b", "Mai"),
            (r"\bJun\b", "Jun"),
            (r"\bJul\b", "Jul"),
            (r"\bAug\b", "Ago"),
            (r"\bSep\b", "Set"),
            (r"\bOct\b", "Out"),
            (r"\bNov\b", "Nov"),
            (r"\bDec\b", "Dez"),
            (r"\bClear sky\b", "Céu limpo"),
            (r"\bSunny intervals\b", "Períodos de céu limpo"),
            (r"\bPartly cloudy\b", "Parcialmente nublado"),
            (r"Cloudy \(High cloud\)", "Nublado (nuvens altas)"),
            (r"\bCloudy\b", "Nublado"),
            (r"\bHigh cloud\b", "nuvens altas"),
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

    text = re.sub(
        r"(?m)^([✅⚠️🟡🟠🔴🌊])\s+\*\*(.*?)\*\*$",
        r"\1 \2",
        text,
    )

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
        # Strip leading emoji + whitespace and a trailing colon, then match the
        # remaining text against the known section titles. Substring matching
        # would misclassify lines like ``✅ Sem avisos meteorológicos ativos.``
        # as a section header just because they contain the words "Avisos
        # Meteorológicos".
        stripped = line.strip().rstrip(":")
        # Drop a single leading emoji cluster (followed by optional VS16) so
        # ``🌤️ Lisbon Weather Summary`` collapses to ``Lisbon Weather Summary``.
        emoji_stripped = re.sub(
            r"^[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s*",
            "",
            stripped,
        ).strip()
        candidate = emoji_stripped.lower()
        return any(candidate == marker.lower() for marker in section_markers)

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

    # Short-circuit: only apply structured nesting when the input actually
    # contains the day/section structure that justifies it. A single short
    # status/detail line (e.g. ``🌤️ Forecast body`` from a fact-check shim)
    # should be returned unchanged so callers do not see a spurious leading
    # ``- `` prefix.
    raw_lines = [line for line in text.splitlines() if line.strip()]
    has_structural_anchor = any(
        _is_section_line(line) or _is_day_line(line) for line in raw_lines
    )
    if not has_structural_anchor:
        return text.strip()

    structured_lines: list[str] = []
    inside_day_block = False
    source_lines = text.splitlines()

    def _peek_next_nonblank_kind(start_idx: int) -> str:
        """Returns the semantic kind of the next non-blank line after start_idx."""
        for j in range(start_idx + 1, len(source_lines)):
            candidate = source_lines[j].strip()
            if not candidate:
                continue
            candidate = re.sub(r"^(?:[-*•]\s+)", "", candidate)
            candidate = _unwrap_full_line_bold(candidate)
            if not candidate:
                continue
            if _is_detail_line(candidate):
                return "detail"
            if _is_day_line(candidate):
                return "day"
            if _is_section_line(candidate):
                return "section"
            if _is_status_line(candidate):
                return "status"
            if _SOURCE_LINE_RE.match(candidate):
                return "source"
            return "other"
        return ""

    for idx, raw_line in enumerate(source_lines):
        stripped = raw_line.strip()
        # A blank line usually indicates a paragraph break. Preserve the
        # ``inside_day_block`` context only when the next non-blank line is
        # still a detail bullet that logically belongs to the previous day.
        # ``format_response`` inserts blanks between every bullet in the
        # formatted weather output, so a blanket reset would strip the
        # indentation from every detail line.
        if not stripped:
            if _peek_next_nonblank_kind(idx) != "detail":
                inside_day_block = False
            continue
        stripped = re.sub(r"^(?:[-*•]\s+)", "", stripped)
        stripped = _unwrap_full_line_bold(stripped)
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
            stripped = _strip_markdown_formatting(stripped)
            prefix = "  - " if inside_day_block else "- "
            structured_lines.append(f"{prefix}{stripped}")
            inside_day_block = False
            continue

        structured_lines.append(stripped)
        inside_day_block = False

    structured = clean_newlines("\n".join(structured_lines)).strip()
    structured = re.sub(
        r"(?m)^\*\*([✅⚠️🟡🟠🔴🌊][^*]+)\*\*$",
        r"\1",
        structured,
    )
    structured = re.sub(
        r"(?m)^\*\*(🌤️\s+(?:As condições meteorológicas são normais|Weather conditions are normal)\.?)\*\*$",
        r"- \1",
        structured,
    )
    return structured.strip()


def _strip_markdown_formatting(text: str) -> str:
    """Remove lightweight markdown emphasis tokens from a text fragment."""
    return re.sub(r"\*\*(.*?)\*\*", r"\1", text or "").strip()


def _normalize_planner_line(text: str) -> str:
    """Remove planner-specific markdown noise before structural parsing."""
    cleaned = _strip_markdown_formatting(text)
    cleaned = re.sub(r"^(?:###\s*)?(?:[-*•]\s*)?(?:#+\s*)?", "", cleaned).strip()
    cleaned = re.sub(r"(\d{1,2})\s*:\s*(\d{2})", r"\1:\2", cleaned)
    cleaned = re.sub(r"\s*[·•]\s*", " · ", cleaned)
    return cleaned


def _is_planner_metadata_line(text: str) -> bool:
    """Detect non-activity planner lines that should not become timed cards."""
    normalized = _strip_accents_compat(_strip_markdown_formatting(text)).lower()
    has_schedule_day = bool(
        re.search(
            r"\b(seg(?:unda)?|terca|terça|quarta|quinta|sexta|sabado|sábado|domingo|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            normalized,
        )
        and re.search(r"\d{1,2}:\d{2}", normalized)
    )
    return has_schedule_day or any(
        keyword in normalized
        for keyword in (
            "horario",
            "opening hours",
            "proximas saidas",
            "next departures",
            "next metros",
            "proximos metros",
            "site oficial",
            "official site",
            "morada",
            "address",
            "coordenad",
            "coord",
            "website",
            "source",
            "fonte",
        )
    )


def _planner_section_icon(label: str) -> str:
    """Pick a user-facing icon for non-timed planner sections."""
    lowered = _strip_markdown_formatting(label).lower()
    if any(keyword in lowered for keyword in ("antes de sair", "before you go", "weather", "meteorolog")):
        return "⛅"
    if any(keyword in lowered for keyword in ("dica", "tip", "nota", "note")):
        return "✨"
    if any(keyword in lowered for keyword in ("transport", "metro", "carris", "cp", "autocarro", "bus")):
        return "🚇"
    return "📝"


def _planner_activity_icon(title: str, emoji: str = "") -> str:
    """Pick an icon for itinerary activities, preserving any existing emoji when possible."""
    if emoji and emoji.strip():
        return emoji.strip()

    lowered = _strip_markdown_formatting(title).lower()
    if any(keyword in lowered for keyword in ("pastel", "nata", "bakery", "pastry")):
        return "🥐"
    if any(keyword in lowered for keyword in ("café", "cafe", "coffee", "aperitivo", "aperitif", "esplanada", "drink")):
        return "☕"
    if any(keyword in lowered for keyword in ("mosteiro", "monastery", "igreja", "church")):
        return "⛪"
    if any(keyword in lowered for keyword in ("museu", "museum", "galeria", "gallery", "arqueologia", "archaeology")):
        return "🏛️"
    if any(keyword in lowered for keyword in ("torre", "tower", "castelo", "castle")):
        return "🏰"
    if any(keyword in lowered for keyword in ("padrão", "padrao", "monument", "descobrimentos", "discoveries")):
        return "🗿"
    if any(keyword in lowered for keyword in ("jardim", "garden", "praça", "praca", "passeio", "walk", "marginal", "tejo", "river")):
        return "🌿"
    if any(keyword in lowered for keyword in ("almoço", "almoco", "lunch", "jantar", "dinner", "restaurant", "restaurante", "comer", "meal")):
        return "🍽️"
    if any(keyword in lowered for keyword in ("transport", "transporte", "metro", "autocarro", "bus")):
        return "🚇"
    return "📍"


def _strip_leading_section_emoji(text: str) -> str:
    """Remove a leading emoji already present in a planner section label."""
    return re.sub(
        r"^[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D\s]+",
        "",
        text or "",
    ).strip()


def _planner_display_heading(text: str, language: str) -> str:
    """Normalize planner headings into consistent PT/EN display title case."""
    cleaned = _strip_leading_section_emoji(text or "").rstrip(":")
    return to_display_title_case(cleaned, language=language)


_PLANNER_CLOCK_EMOJI_TO_TIME = {
    "🕐": (1, 0),
    "🕜": (1, 30),
    "🕑": (2, 0),
    "🕝": (2, 30),
    "🕒": (3, 0),
    "🕞": (3, 30),
    "🕓": (4, 0),
    "🕟": (4, 30),
    "🕔": (5, 0),
    "🕠": (5, 30),
    "🕕": (6, 0),
    "🕖": (7, 0),
    "🕗": (8, 0),
    "🕘": (9, 0),
    "🕙": (10, 0),
    "🕚": (11, 0),
    "🕛": (12, 0),
}


def _planner_clock_to_time(clock_emoji: str, afternoon_context: bool) -> Optional[str]:
    """Convert a clock-face emoji into a readable HH:MM slot when possible."""
    clock_value = _PLANNER_CLOCK_EMOJI_TO_TIME.get(clock_emoji)
    if not clock_value:
        return None

    hour, minute = clock_value
    if afternoon_context and 1 <= hour <= 6:
        hour += 12
    return f"{hour:02d}:{minute:02d}"


def structure_planner_markdown(text: str) -> str:
    """
    Enforces the premium card layout for itineraries by transforming
    flat text into a visual card structure with horizontal rules.
    """
    if not text:
        return text

    language = infer_response_language(context_text=text, default="en")
    structured: list[str] = []
    current_block: Optional[str] = None
    overall_title_rendered = False
    afternoon_context = bool(re.search(r"\b(tarde|afternoon)\b", text, re.IGNORECASE))
    seen_section_headings: set[str] = set()

    def append_separator() -> None:
        if not structured:
            return
        while structured and not structured[-1].strip():
            structured.pop()
        if structured and structured[-1] != "---":
            structured.extend(["", "---", ""])

    def append_semantic_section(heading: str) -> None:
        nonlocal current_block
        if heading in seen_section_headings:
            current_block = "section"
            return
        append_separator()
        structured.append(heading)
        seen_section_headings.add(heading)
        current_block = "section"

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped == "---":
            continue

        lowered_stripped = stripped.lower()
        if lowered_stripped.startswith(("**fontes citadas**", "fontes citadas", "**sources cited**", "sources cited")):
            continue

        if _SOURCE_LINE_RE.match(stripped):
            append_separator()
            structured.append(stripped)
            current_block = None
            continue

        normalized = _normalize_planner_line(stripped)
        lowered = normalized.lower()
        if not normalized:
            continue

        title_window_match = re.search(r"(\d{1,2}:\d{2}\s*[→-]\s*\d{1,2}:\d{2})", normalized)

        if any(
            keyword in lowered
            for keyword in (
                "condições e segurança",
                "condicoes e seguranca",
                "weather and safety",
                "conditions and safety",
            )
        ) and ":" not in normalized:
            append_semantic_section(f"### ⛅ {_planner_display_heading(normalized, language)}")
            continue

        if re.search(r"\b(como chegar|desloca(?:r-se|ção)|how to get there|get around)\b", lowered) and ":" not in normalized:
            append_semantic_section(f"### 🚇 {_planner_display_heading(normalized, language)}")
            continue

        if any(
            keyword in lowered
            for keyword in (
                "sugestões para a visita",
                "sugestoes para a visita",
                "sugestões",
                "sugestoes",
                "recomendações",
                "recomendacoes",
                "recommendations",
                "opções",
                "opcoes",
                "options",
                "visit suggestions",
                "para a visita",
            )
        ) and ":" not in normalized:
            append_semantic_section(f"### 📍 {_planner_display_heading(normalized, language)}")
            continue

        if re.search(r"\b(fontes|verificaç|verification|sources?)\b", lowered) and ":" not in normalized:
            append_semantic_section(f"### 🔎 {_planner_display_heading(normalized, language)}")
            continue

        if re.search(
            r"\b(hor[aá]rio indicado|opening hours?|pode j[aá] estar encerrado|may already be closed)\b",
            lowered,
        ):
            if not current_block:
                append_separator()
                heading = "Notas Importantes" if language == "pt" else "Important Notes"
                structured.append(f"### ⚠️ {_planner_display_heading(heading, language)}")
                current_block = "section"
            structured.append(f"- ⚠️ {normalized.rstrip('.')}.")
            continue

        activity_match = re.match(
            r"^(?P<emoji>[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+)?\s*(?P<time>\d{1,2}:\d{2})\s*[-–—:]\s*(?P<title>.+)$",
            normalized,
        )
        if activity_match and "atualizado" not in lowered and "updated" not in lowered:
            title = activity_match.group("title").strip(" -–—")
            if _is_planner_metadata_line(title):
                normalized = " ".join(
                    part
                    for part in ((activity_match.group("emoji") or "").strip(), title)
                    if part
                ).strip()
                lowered = normalized.lower()
            else:
                append_separator()
                icon = _planner_activity_icon(title, activity_match.group("emoji") or "")
                structured.append(f"### {icon} {activity_match.group('time')} · {title}")
                current_block = "activity"
                continue

        clock_activity_match = re.match(
            r"^(?P<clock>[🕐🕜🕑🕝🕒🕞🕓🕟🕔🕠🕕🕖🕗🕘🕙🕚🕛])\s*(?P<title>.+)$",
            normalized,
        )
        if clock_activity_match:
            derived_time = _planner_clock_to_time(
                clock_activity_match.group("clock"),
                afternoon_context=afternoon_context,
            )
            if derived_time:
                title = clock_activity_match.group("title").strip(" -–—")
                if _is_planner_metadata_line(title):
                    structured.append(f"- {clock_activity_match.group('clock')} {title}")
                    current_block = current_block or "section"
                    continue
                append_separator()
                icon = _planner_activity_icon(title)
                structured.append(f"### {icon} {derived_time} · {title}")
                current_block = "activity"
                continue

        enumerated_item_match = re.match(
            r"^(?P<num>\d+)[\.\)]\s+(?P<title>.+)$",
            normalized,
        )
        if enumerated_item_match:
            title = enumerated_item_match.group("title").strip(" -–—")
            if _is_planner_metadata_line(title):
                structured.append(f"- {title}")
                current_block = current_block or "section"
                continue
            append_separator()
            icon = _planner_activity_icon(title)
            structured.append(f"### {icon} {title}")
            current_block = "activity"
            continue

        calendar_window_match = re.search(
            r"^(?P<emoji>📅)\s*(?P<label>.+?)(?:\s*[:,]\s*|\s+)(?P<window>\d{1,2}:\d{2}\s*(?:[–—−‑-]|to)\s*\d{1,2}:\d{2})$",
            normalized,
            flags=re.IGNORECASE,
        )
        if calendar_window_match and not overall_title_rendered:
            clean_title = calendar_window_match.group("label").strip().rstrip(",:- ")
            window_value = re.sub(
                r"\s*(?:(?P<dash>[–—−‑-])|(?P<word>to))\s*",
                lambda match: match.group("dash") or " to ",
                calendar_window_match.group("window").strip(),
                flags=re.IGNORECASE,
            )
            structured.append(f"### 📅 {to_display_title_case(clean_title, language=language)}")
            structured.append(
                f"⏰ **{'Janela sugerida:' if language == 'pt' else 'Suggested window:'}** {window_value}"
            )
            overall_title_rendered = True
            current_block = "section"
            continue

        if (
            not overall_title_rendered
            and re.search(r"\b(itinerário|itinerary|plano|roteiro)\b", lowered)
        ):
            clean_title = re.sub(
                r"^[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s*",
                "",
                normalized,
            ).rstrip(":")
            if title_window_match:
                clean_title = re.sub(r"\s*\([^)]*\d{1,2}:\d{2}[^)]*\)", "", clean_title).strip()
            structured.append(f"### 📅 {to_display_title_case(clean_title, language=language)}")
            if title_window_match:
                structured.append(
                    f"- ⏰ **Janela sugerida**: {title_window_match.group(1)}"
                )
            overall_title_rendered = True
            current_block = "section"
            continue

        preface_match = re.match(
            r"^(?P<label>Antes de sair|Before you go)\s*,\s*(?P<content>.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if preface_match:
            label = preface_match.group("label")
            append_semantic_section(f"### {_planner_section_icon(label)} {_planner_display_heading(label, language)}")
            structured.append(f"- {preface_match.group('content').strip()}")
            continue

        if any(
            keyword in lowered
            for keyword in (
                "dicas práticas",
                "dicas praticas",
                "practical tips",
                "important notes",
                "notas importantes",
                "notas práticas",
                "notas praticas",
                "final notes",
            )
        ) and ":" not in normalized:
            append_semantic_section(f"### ✨ {_planner_display_heading(normalized, language)}")
            continue

        section_match = re.match(
            r"^(?P<emoji>[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+)?\s*(?P<label>[^:]{2,60})\s*:\s*(?P<content>.+)$",
            normalized,
        )
        if section_match:
            label = section_match.group("label").strip().rstrip("-–—")
            content = section_match.group("content").strip().rstrip(",;")
            label_lower = label.lower()
            is_major_section = any(
                keyword in label_lower
                for keyword in (
                    "antes de sair",
                    "before you go",
                    "dicas práticas",
                    "practical tips",
                    "important notes",
                    "notas importantes",
                )
            )
            if is_major_section:
                append_semantic_section(f"### {_planner_section_icon(label)} {_planner_display_heading(label, language)}")
                if content:
                    structured.append(f"- {content}")
                continue

            bullet_icon = (section_match.group("emoji") or "").strip() or "🔹"
            structured.append(f"- {bullet_icon} **{to_display_title_case(label, language=language)}**: {content}")
            current_block = current_block or "section"
            continue

        poi_heading_match = re.match(
            r"^(?P<emoji>[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+)\s+(?P<title>[A-Za-zÀ-ÿ].+)$",
            normalized,
        )
        if (
            poi_heading_match
            and poi_heading_match.group("emoji").strip() not in {"⛅", "🚇", "📍", "🔎", "✨", "⚠️", "📝"}
            and ":" not in normalized
        ):
            structured.append(
                f"- {poi_heading_match.group('emoji').strip()} **{poi_heading_match.group('title').strip()}**"
            )
            current_block = current_block or "section"
            continue

        bullet_content = re.sub(r"^(?:[-*•]\s*)", "", normalized).strip()
        if current_block:
            structured.append(f"- {bullet_content}")
        else:
            structured.append(bullet_content)

    return clean_newlines("\n".join(structured)).strip()


def soften_internal_markdown_headers(
    text: str,
    *,
    preserve_first_header: bool = True,
    preserve_timed_cards: bool = True,
) -> str:
    """Convert internal markdown headers into softer section labels.

    This keeps the main response title and timed itinerary cards intact while
    making the remaining sections feel closer to the cleaner weather/event UI.
    """
    if not text:
        return text

    language = infer_response_language(context_text=text, default="en")
    softened_lines: list[str] = []
    header_count = 0

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        header_match = re.match(r"^(#{3,4})\s+(.+)$", stripped)
        if not header_match:
            softened_lines.append(raw_line)
            continue

        header_count += 1
        title = header_match.group(2).strip()
        plain_title = _strip_markdown_formatting(title)

        if preserve_first_header and header_count == 1 and len(header_match.group(1)) == 3:
            softened_lines.append(stripped)
            continue

        if preserve_timed_cards and _TIMED_SECTION_HEADER_RE.match(plain_title):
            softened_lines.append(stripped)
            continue

        if softened_lines and softened_lines[-1].strip():
            softened_lines.append("")
        softened_lines.append(f"**{to_display_title_case(title, language=language)}**")

    return clean_newlines("\n".join(softened_lines)).strip()


def _looks_like_pt_transport_text(text: str) -> bool:
    """Infer whether a transport response is primarily in PT-PT."""
    return bool(
        re.search(
            r"\b(pr[oó]xim(?:as|os)|chegadas|destino|paragens|hor[aá]rio|atualizado|fonte|dica|autocarros?)\b",
            text or "",
            re.IGNORECASE,
        )
    )


def _clean_transport_arrival_title(title: str, is_pt: bool) -> str:
    """Normalize Carris arrival titles into a concise H3 heading."""
    plain = _strip_markdown_formatting(title)
    plain = re.sub(r"\((?:paragem|stop).*?\)", "", plain, flags=re.IGNORECASE).strip()
    plain = re.sub(r"^(?:🚌|🚋|🚇|🚆)\s*", "", plain).strip()

    if re.match(r"^Pr[oó]ximas\s+Chegadas\s*:\s*", plain, flags=re.IGNORECASE):
        stop_name = re.sub(r"^Pr[oó]ximas\s+Chegadas\s*:\s*", "", plain, flags=re.IGNORECASE).strip()
        return f"### 🚌 {stop_name} · Próximas Chegadas"

    if re.match(r"^Next\s+Arrivals?\s*:\s*", plain, flags=re.IGNORECASE):
        stop_name = re.sub(r"^Next\s+Arrivals?\s*:\s*", "", plain, flags=re.IGNORECASE).strip()
        return f"### 🚌 {stop_name} · Next Arrivals"

    if "→" in plain:
        plain = re.sub(
            r"\s*→\s*(Pr[oó]ximas\s+chegadas|Next\s+Arrivals?)",
            lambda match: f" · {to_display_title_case(match.group(1), language='pt' if is_pt else 'en')}",
            plain,
            flags=re.IGNORECASE,
        )

    return f"### 🚌 {plain}" if plain else ("### 🚌 Próximas Chegadas" if is_pt else "### 🚌 Next Arrivals")


def _build_carris_source_line(is_pt: bool, timestamp: Optional[str]) -> Optional[str]:
    """Build a canonical Carris source line when only a timestamp is available."""
    if not timestamp:
        return None
    if is_pt:
        return f"📌 **Fonte:** [*Carris*](https://www.carris.pt) | **Atualizado:** {timestamp}"
    return f"📌 **Source:** [*Carris*](https://www.carris.pt) | **Updated:** {timestamp}"


def _compact_transport_arrivals_markdown(text: str) -> Optional[str]:
    """Compact Carris arrival summaries into grouped real-time and scheduled sections."""
    if not text:
        return None

    is_pt = _looks_like_pt_transport_text(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    entry_header_re = re.compile(
        r"^(?:[-*•]\s*)?(?P<emoji>[🚌🚋🚇🚆])\s*(?:\*\*)?(?P<line>[0-9A-Z]{1,5})(?:\*\*)?\s*[-–—]\s*(?:(?:\*\*)?(?:Destino|Destination)(?:\*\*)?\s*:\s*)?(?P<destination>.+)$",
        re.IGNORECASE,
    )
    alternate_header_re = re.compile(
        r"^\[(?P<status>REAL-TIME|Hor[áa]rio|Scheduled)\]\s+(?P<mode>Autocarro|Bus|El[eé]trico|Tram)\s+(?P<line>[0-9A-Z]{1,5})\s*->\s*(?P<destination>.+)$",
        re.IGNORECASE,
    )

    title_line: Optional[str] = None
    source_line: Optional[str] = None
    source_timestamp: Optional[str] = None
    notes: list[str] = []
    entries: list[dict[str, object]] = []
    current_entry: Optional[dict[str, object]] = None

    for line in lines:
        if _SOURCE_LINE_RE.match(line):
            source_line = line
            source_timestamp = extract_update_time(line) or source_timestamp
            current_entry = None
            continue

        timestamp_from_line = extract_update_time(line)
        if timestamp_from_line:
            source_timestamp = timestamp_from_line
            current_entry = None
            continue

        if "GTFS-RT" in line or "cached live snapshot" in line.lower():
            current_entry = None
            continue

        plain_line = _strip_markdown_formatting(line)
        if plain_line.startswith(("💡", "ℹ️")) or re.match(r"^(?:Quick tip|Dica rápida)", plain_line, flags=re.IGNORECASE):
            if re.search(r"ve[ií]culos? identificados?|vehicle ids?|matr[íi]culas?", plain_line, flags=re.IGNORECASE):
                notes.append(
                    "💡 **Dica rápida:** Os tempos assinalados como em tempo real usam dados GPS recentes da Carris."
                    if is_pt
                    else "💡 **Quick tip:** Real-time labels use recent Carris GPS data."
                )
            else:
                notes.append(plain_line)
            current_entry = None
            continue

        if re.match(r"^\[(?:REAL-TIME|Hor[áa]rio|Scheduled)\]\s*=", plain_line, flags=re.IGNORECASE):
            notes.append(
                "💡 **Dica rápida:** “Em tempo real” usa dados GPS recentes; os restantes horários são programados."
                if is_pt
                else "💡 **Quick tip:** “Real time” uses recent GPS data, while the remaining times are scheduled."
            )
            current_entry = None
            continue

        header_match = entry_header_re.match(plain_line)
        if header_match:
            current_entry = {
                "emoji": header_match.group("emoji"),
                "line": header_match.group("line"),
                "destination": header_match.group("destination").strip(),
                "time": "",
                "live": False,
                "scheduled": False,
                "extras": [],
            }
            entries.append(current_entry)
            continue

        alternate_match = alternate_header_re.match(plain_line)
        if alternate_match:
            mode = alternate_match.group("mode").lower()
            status = alternate_match.group("status").lower()
            current_entry = {
                "emoji": "🚋" if any(token in mode for token in ("elétrico", "eletrico", "tram")) else "🚌",
                "line": alternate_match.group("line"),
                "destination": alternate_match.group("destination").strip(),
                "time": "",
                "live": "real-time" in status,
                "scheduled": "hor" in status or "scheduled" in status,
                "extras": [],
            }
            entries.append(current_entry)
            continue

        if title_line is None and not entries:
            title_line = line
            continue

        if current_entry is None:
            continue

        time_match = re.search(r"(?P<time>\d{1,2}:\d{2})", plain_line)
        if time_match:
            current_entry["time"] = time_match.group("time")

        if re.search(r"em tempo real|real[- ]time", plain_line, flags=re.IGNORECASE):
            current_entry["live"] = True
        if re.search(r"hor[áa]rio|scheduled", plain_line, flags=re.IGNORECASE):
            current_entry["scheduled"] = True

        extras = plain_line
        extras = re.sub(r"^[🕒⏱️\s-]+", "", extras)
        extras = re.sub(r"^(?:Hora|Time)\s*:\s*", "", extras, flags=re.IGNORECASE)
        if time_match:
            extras = extras.replace(time_match.group("time"), "", 1)
        extras = re.sub(
            r"[—-]?\s*(Em tempo real(?:\s*\([^)]*\))?|Real[- ]time(?:\s*\([^)]*\))?|Hor[áa]rio(?: programado)?|Scheduled(?:\s+times?)?)",
            "",
            extras,
            flags=re.IGNORECASE,
        )
        if is_pt:
            extras = re.sub(r"(\d+)\s+min\s+late", r"atraso \1 min", extras, flags=re.IGNORECASE)
            extras = re.sub(r"(\d+)\s+stops?\s+remaining", r"\1 paragens restantes", extras, flags=re.IGNORECASE)
        extras = extras.strip(" ()—-·;,")
        extras = re.sub(r"[ \t]{2,}", " ", extras)
        if extras:
            extras_list = current_entry.get("extras")
            if not isinstance(extras_list, list):
                extras_list = []
                current_entry["extras"] = extras_list
            if extras not in extras_list:
                extras_list.append(extras)

    if not entries:
        return None

    def format_entry(entry: dict[str, object]) -> str:
        parts = [
            f"- {entry['emoji']} **{entry['line']}** → {entry['destination']}",
        ]
        if entry.get("time"):
            parts.append(f"**{entry['time']}**")
        raw_extras = entry.get("extras", [])
        extras = raw_extras if isinstance(raw_extras, list) else []
        parts.extend(str(item) for item in extras if item)
        return " · ".join(parts)

    realtime_entries = [entry for entry in entries if entry.get("live") and not entry.get("scheduled")]
    scheduled_entries = [entry for entry in entries if entry not in realtime_entries]

    output_lines = [
        _clean_transport_arrival_title(title_line or "", is_pt),
    ]

    if realtime_entries:
        output_lines.extend([
            "",
            "**Em tempo real**" if is_pt else "**Real time**",
            *[format_entry(entry) for entry in realtime_entries],
        ])

    if scheduled_entries:
        output_lines.extend([
            "",
            "**Horários programados**" if is_pt else "**Scheduled times**",
            *[format_entry(entry) for entry in scheduled_entries],
        ])

    if notes:
        output_lines.extend(["", *notes])

    source_line = source_line or _build_carris_source_line(is_pt, source_timestamp)
    if source_line:
        output_lines.extend(["", source_line])

    return clean_newlines("\n".join(output_lines)).strip()


def _structure_transport_route_dump_markdown(text: str) -> Optional[str]:
    """Convert raw Carris route dumps into a cleaner, card-like markdown layout."""
    if not text:
        return None

    if not re.search(r"^\s*\*{0,2}Routes\*{0,2}\s*:", text, re.IGNORECASE | re.MULTILINE):
        return None

    lines = [line.rstrip() for line in text.splitlines()]
    is_pt = _looks_like_pt_transport_text(text)

    route_title_re = re.compile(
        r"^\s*\*{0,2}Routes\*{0,2}\s*:\s*(?P<origin>.+?)\s*(?:->|→)\s*(?P<destination>.+?)\s*$",
        re.IGNORECASE,
    )
    mode_heading_re = re.compile(r"^(BUSES|TRAMS|TRAINS|METRO)\s*$", re.IGNORECASE)
    route_line_re = re.compile(r"^(?P<line>[0-9A-Z]{1,6}[A-Z]?)\s*:\s*(?P<destination>.+)$")
    resolved_from_re = re.compile(r"^\*{0,2}From\*{0,2}\s*:\s*(?P<value>.+)$", re.IGNORECASE)
    resolved_to_re = re.compile(r"^\*{0,2}To\*{0,2}\s*:\s*(?P<value>.+)$", re.IGNORECASE)
    count_re = re.compile(r"Found\s+(?P<count>\d+)\s+direct\s+routes?!", re.IGNORECASE)

    summary: dict[str, str] = {}
    sections: dict[str, list[dict[str, object]]] = {}
    current_mode: Optional[str] = None
    current_entry: Optional[dict[str, object]] = None

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or set(stripped) <= {"=", "-", " "}:
            continue

        route_title_match = route_title_re.match(stripped)
        if route_title_match:
            summary["origin"] = route_title_match.group("origin").strip()
            summary["destination"] = route_title_match.group("destination").strip()
            continue

        from_match = resolved_from_re.match(stripped)
        if from_match:
            summary["resolved_origin"] = from_match.group("value").strip()
            continue

        to_match = resolved_to_re.match(stripped)
        if to_match:
            summary["resolved_destination"] = to_match.group("value").strip()
            continue

        count_match = count_re.search(stripped)
        if count_match:
            summary["direct_count"] = count_match.group("count")
            continue

        if "GTFS-RT" in stripped.upper():
            summary["feed_status"] = _strip_markdown_formatting(stripped)
            continue

        mode_match = mode_heading_re.match(stripped)
        if mode_match:
            current_mode = mode_match.group(1).upper()
            sections.setdefault(current_mode, [])
            current_entry = None
            continue

        route_match = route_line_re.match(_strip_markdown_formatting(stripped))
        if current_mode and route_match:
            current_entry = {
                "line": route_match.group("line").strip(),
                "destination": route_match.group("destination").strip(),
                "notes": [],
            }
            sections[current_mode].append(current_entry)
            continue

        if current_entry is None:
            continue

        normalized = _strip_markdown_formatting(stripped)
        if re.match(r"^Next\s*:", normalized, re.IGNORECASE):
            current_entry["next"] = re.sub(r"^Next\s*:\s*", "", normalized, flags=re.IGNORECASE).strip()
        elif "tempo real" in normalized.lower() or "real-time" in normalized.lower():
            current_entry["realtime"] = normalized
        elif "travel" in normalized.lower() or "viagem" in normalized.lower():
            current_entry["travel_time"] = normalized
        else:
            notes = current_entry.setdefault("notes", [])
            if isinstance(notes, list):
                notes.append(normalized)

    if not sections:
        return None

    mode_titles = {
        "pt": {"BUSES": "#### 🚌 Autocarros", "TRAMS": "#### 🚋 Elétricos", "TRAINS": "#### 🚆 Comboios", "METRO": "#### 🚇 Metro"},
        "en": {"BUSES": "#### 🚌 Buses", "TRAMS": "#### 🚋 Trams", "TRAINS": "#### 🚆 Trains", "METRO": "#### 🚇 Metro"},
    }
    mode_icons = {"BUSES": "🚌", "TRAMS": "🚋", "TRAINS": "🚆", "METRO": "🚇"}
    language_key = "pt" if is_pt else "en"
    output_lines: list[str] = []

    origin_display = summary.get("resolved_origin") or summary.get("origin")
    destination_display = summary.get("resolved_destination") or summary.get("destination")
    if origin_display and destination_display:
        output_lines.append(
            f"**Trajeto:** {origin_display} → {destination_display}"
            if is_pt
            else f"**Route:** {origin_display} → {destination_display}"
        )
        output_lines.append("")

    if summary.get("direct_count"):
        output_lines.append(
            f"📊 **Ligações diretas encontradas:** {summary['direct_count']}"
            if is_pt
            else f"📊 **Direct connections found:** {summary['direct_count']}"
        )
    if summary.get("feed_status"):
        feed_status = str(summary["feed_status"])
        if is_pt:
            feed_status = feed_status.replace("live active", "tempo real ativo")
            output_lines.append(f"📡 **Tempo real:** {feed_status}")
        else:
            output_lines.append(f"📡 **Real time:** {feed_status}")
    if output_lines and output_lines[-1] != "":
        output_lines.append("")

    for mode_name in ("METRO", "TRAINS", "TRAMS", "BUSES"):
        entries = sections.get(mode_name, [])
        if not entries:
            continue
        output_lines.extend([mode_titles[language_key][mode_name], ""])
        for item_number, entry in enumerate(entries, 1):
            line_label = "Linha" if is_pt else "Line"
            departures_label = "Próximas saídas" if is_pt else "Next departures"
            realtime_label = "Tempo real" if is_pt else "Real time"
            travel_label = "Tempo estimado" if is_pt else "Estimated travel time"
            note_label = "Nota" if is_pt else "Note"
            icon = mode_icons.get(mode_name, "🚌")

            output_lines.append(
                f"{item_number}. {icon} **{line_label} {entry.get('line', '')}** — {entry.get('destination', '')}"
            )
            if entry.get("next"):
                output_lines.append(f"   🕐 **{departures_label}:** {entry['next']}")
            if entry.get("realtime"):
                output_lines.append(f"   ℹ️ **{realtime_label}:** {entry['realtime']}")
            if entry.get("travel_time"):
                travel_value = str(entry["travel_time"]).replace("~", "~ ").replace("min travel", "min")
                output_lines.append(f"   ⏱️ **{travel_label}:** {travel_value.strip()}")
            notes = entry.get("notes", [])
            if isinstance(notes, list):
                for note in notes:
                    output_lines.append(f"   ℹ️ **{note_label}:** {note}")
            output_lines.append("")

    structured = clean_newlines("\n".join(output_lines)).strip()
    if not structured:
        return None

    if not has_source_line(structured):
        structured = f"{structured}\n\n{_build_carris_source_line(is_pt, datetime.now().strftime('%H:%M'))}"
    return structured


def structure_transport_markdown(text: str) -> str:
    """
    Cleans up transport agent text, normalizing placeholders and improving
    readability for raw route dumps and stop schedules.
    """
    if not text:
        return text

    placeholder_patterns = [
        r"\[Check schedule\]", r"\(Check schedule\)", r'["\']?Check schedule["\']?',
        r"\[Tempo \d+\]", r"\[Time \d+\]",
        r"\[Nº Linha\]", r"\[Destino\]", r"\[Tempos\]",
        r"\[Origin\]", r"\[Destination\]", r"\[Station\]", r"\[Direction\]",
        r"\[Transfer Station\]", r"\[Landmark\]", r"\[Name\]",
    ]
    for pattern in placeholder_patterns:
        text = re.sub(pattern, "(Sem informação em tempo real)", text, flags=re.IGNORECASE)

    route_dump = _structure_transport_route_dump_markdown(text)
    if route_dump:
        return route_dump.strip()

    compacted = _compact_transport_arrivals_markdown(text)
    if compacted:
        return compacted.strip()

    text = re.sub(r"^(?:\s*-\s*)?([0-9A-Z]{2,4})\s*-\s*", r"- 🚌 **\1** - ", text, flags=re.MULTILINE)
    text = re.sub(r"\bHorario\b", "Horário", text, flags=re.IGNORECASE)

    return clean_newlines(text).strip()


def strip_transport_weather_disclaimers(text: str) -> str:
    """Removes weather-side disclaimers that sometimes leak into transport answers."""
    if not text:
        return text

    cleaned = _TRANSPORT_WEATHER_BLOCK_RE.sub("\n", text)

    result_lines: list[str] = []
    skipping_weather_block = False
    restart_markers = (
        "**Opção",
        "**Option",
        "**Horários programados**",
        "**Scheduled times**",
        "### ",
        "🚇",
        "🚌",
        "🚆",
        "📌",
    )

    for raw_line in cleaned.splitlines():
        stripped = raw_line.strip()
        lowered = stripped.lower()

        if re.search(r"(?:sobre o tempo em lisboa|tempo em lisboa|weather in lisbon|weather update)", lowered):
            skipping_weather_block = True
            continue

        if skipping_weather_block:
            if stripped.startswith(restart_markers) or "como ir do" in lowered or "how to get from" in lowered:
                skipping_weather_block = False
            else:
                continue

        if re.search(
            r"n[aã]o tenho acesso a dados meteorol[oó]gicos|don't have access to (?:real-time )?weather|google weather|in-weather",
            lowered,
        ):
            continue
        if re.search(r"recomend[oa].*(?:ipma|weather|previs|forecast)", lowered):
            continue

        result_lines.append(raw_line)

    return clean_newlines("\n".join(result_lines)).strip()


def canonicalize_transport_terms(text: str, language: str = "en") -> str:
    """Normalizes transport-summary labels for the requested response language."""
    if not text:
        return text

    if language == "en":
        replacements = [
            (r"Situação dos Transportes de Lisboa", "Lisbon Transport Status"),
            (r"\bAtualizado:\b", "Updated:"),
            (r"\bAtualizado às\b", "Updated at"),
            (r"\*\*Estado\*\*:", "**Status**:"),
            (r"\*\*Estado das Linhas:\*\*", "**Line Status:**"),
            (r"\*\*Comboio:", "**Train:"),
            (r"\*\*RESUMO DA VIAGEM\*\*", "**TRIP SUMMARY**"),
            (r"Linha:", "Line:"),
            (r"(\d+(?:-\d+)?)\s+minutos\b", r"\1 min"),
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
    else:
        replacements = [
            (r"Lisbon Transport Status", "Situação dos Transportes de Lisboa"),
            (r"\*\*Route:\s*([^*]+)\*\*", r"**Trajeto:** \1"),
            (r"\*\*Route:\*\*", "**Trajeto:**"),
            (r"\*\*Routes:\*\*", "**Trajetos:**"),
            (r"\*\*(?:LOCATION INFORMATION|Localização INFORMATION)\*\*", "**Informação de localização**"),
            (r"\*\*METRO ROUTE\*\*", "**Percurso de metro**"),
            (r"\*\*Full Route\*\*", "**Percurso completo**"),
            (r"\*\*Transfer Required\*\*", "**É necessária transferência**"),
            (r"\*\*Updated\*\*:", "**Atualizado**:"),
            (r"\*\*Updated:\*\*", "**Atualizado:**"),
            (r"\*\*Source\*\*:", "**Fonte**:"),
            (r"\*\*Source:\*\*", "**Fonte:**"),
            (r"\*\*Quick tip\*\*:", "**Dica rápida**:"),
            (r"\*\*Quick tip:\*\*", "**Dica rápida:**"),
            (r"\*\*Quick Tip\*\*:", "**Dica rápida**:"),
            (r"\*\*Quick Tip:\*\*", "**Dica rápida:**"),
            (r"Updated:", "Atualizado:"),
            (r"Source:", "Fonte:"),
            (r"\bQuick tip:\b", "Dica rápida:"),
            (r"\bQuick Tip:\b", "Dica rápida:"),
            (r"\bActualizado\b", "Atualizado"),
            (r"\bactivo\b", "ativo"),
            (r"\*\*Status\*\*:", "**Estado**:"),
            (r"Status:", "Estado:"),
            (r"\*\*Vehicles in service\*\*:", "**Veículos em serviço**:"),
            (r"Vehicles in service:", "Veículos em serviço:"),
            (r"\*\*Active alerts\*\*:", "**Alertas ativos**:"),
            (r"Active alerts:", "Alertas ativos:"),
            (r"\*\*Trains running in AML\*\*:", "**Comboios a circular na AML**:"),
            (r"Trains running in AML:", "Comboios a circular na AML:"),
            (r"\*\*Trains with delays over 1 minute\*\*:", "**Comboios com atrasos superiores a 1 minuto**:"),
            (r"Trains with delays over 1 minute:", "Comboios com atrasos superiores a 1 minuto:"),
            (r"\*\*Trains with delays > 1 min\*\*:", "**Comboios com atrasos > 1 min**:"),
            (r"Trains with delays > 1 min:", "Comboios com atrasos > 1 min:"),
            (r"\*\*Carris \(Urban\)\*\*", "**Carris (Urbano)**"),
            (r"\*\*Carris \(Urban buses\)\*\*", "**Carris (Urbano)**"),
            (r"Carris \(Urban buses\)", "Carris (Urbano)"),
            (r"\*\*Carris Metropolitana \(Suburban\)\*\*", "**Carris Metropolitana (Suburbano)**"),
            (r"\*\*Carris Metropolitana \(Suburban buses\)\*\*", "**Carris Metropolitana (Suburbano)**"),
            (r"Carris Metropolitana \(Suburban buses\)", "Carris Metropolitana (Suburbano)"),
            (r"\*\*CP trains \(AML\)\*\*", "**CP Comboios (AML)**"),
            (r"CP trains \(AML\)", "CP Comboios (AML)"),
            (r"\bNormal service on all lines\b", "Circulação normal em todas as linhas"),
            (r"\bHelpful Notes\b", "Notas Úteis"),
            (r"(\d+)\s+vehicles\b", r"\1 veículos"),
            (r"(\d+)\s+alerts\b", r"\1 alertas"),
            (r"(\d+)\s+trains\b", r"\1 comboios"),
            (r"\baare\b", "are"),
            (r"\bppodem\b", "podem"),
            (r"Carris Metropolitana has active alerts, but the nature of the disruptions and the affected routes .*?here\.", "A Carris Metropolitana tem alertas ativos, mas a natureza das perturbações e as rotas afetadas não estão especificadas aqui."),
            (r"(?:[-*•]\s*)?The specific affected routes are not listed, so the current operational impact should be verified before traveling\.?", "As rotas especificamente afetadas não estão listadas, por isso o impacto operacional atual deve ser confirmado antes de viajar."),
            (r"(?:[-*•]\s*)?The affected lines, stations, or connections are (?:not specified|Não especificado), so the disruption details should be verified\.?", "As linhas, estações ou ligações afetadas não estão especificadas, por isso os detalhes da perturbação devem ser confirmados."),
            (r"(?:[-*•]\s*)?The available data does not specify which routes are affected or the exact disruption details, so this should be verified\.?", "Os dados disponíveis não especificam quais as rotas afetadas nem os detalhes exatos da perturbação, por isso esta informação deve ser confirmada."),
            (r"(?:[-*•]\s*)?The available data does not specify the affected lines, directions, or transfer points, so this should be verified\.?", "Os dados disponíveis não especificam as linhas, direções ou pontos de transbordo afetados, por isso esta informação deve ser confirmada."),
            (r"(?:[-*•]\s*)?Carris Metropolitana has active alerts, but the impact on specific routes is (?:not specified|Não especificado)\.?", "A Carris Metropolitana tem alertas ativos, mas o impacto em rotas específicas não está especificado."),
            (r"(?:[-*•]\s*)?CP shows delays on some trains in AML, but affected lines or stations are not listed\.?", "A CP apresenta atrasos em alguns comboios na AML, mas as linhas ou estações afetadas não estão listadas."),
            (r"(?:[-*•]\s*)?The source list is incomplete for the full transport picture; only Metro de Lisboa is cited explicitly\.?", "A lista de fontes está incompleta para o panorama total dos transportes; apenas o Metro de Lisboa é citado explicitamente."),
            (r"(?:[-*•]\s*)?The Carris Metropolitana alert count and CP delay counts are not enough to describe the actual disruption status without affected lines/routes or service details\.?", "A contagem de alertas da Carris Metropolitana e os atrasos da CP não chegam para descrever o estado real das perturbações sem linhas, rotas ou detalhes de serviço afetados."),
            (r"(?:[-*•]\s*)?Carris bus route numbers and schedules should be confirmed at carris\.pt, because GTFS data may miss very recent changes\.?", "Os números das linhas e os horários da Carris devem ser confirmados em carris.pt, porque os dados GTFS podem falhar alterações muito recentes."),
            (r"Carris route numbers and schedules should be verified at carris\.pt, as GTFS data may not reflect the most recent changes\.", "Os números de linha e horários da Carris devem ser confirmados em carris.pt, porque os dados GTFS podem não refletir as alterações mais recentes."),
            (r"Carris route numbers and schedules should be confirmed at carris\.pt, as GTFS data may not reflect the+e? most recent changes\.", "Os números de linha e horários da Carris devem ser confirmados em carris.pt, porque os dados GTFS podem não refletir as alterações mais recentes."),
            (r"\bpoddem\b", "podem"),
            (r"\blistaddas\b", "listadas"),
            (r"\bNearest Metro\b", "Metro mais próximo"),
            (r"\bResolved dynamically via OpenStreetMap/Nominatim\b", "Resolvido dinamicamente via OpenStreetMap/Nominatim"),
            (r"\*\*Direct connections found:\*\*", "**Ligações diretas encontradas:**"),
            (r"\bDirect connections found:\b", "Ligações diretas encontradas:"),
            (r"\*\*🚌\s*Buses\*\*", "**🚌 Autocarros**"),
            (r"\*\*🚋\s*Trams\*\*", "**🚋 Elétricos**"),
            (r"\*\*🚆\s*Trains\*\*", "**🚆 Comboios**"),
            (r"\*\*Buses\*\*", "**Autocarros**"),
            (r"\*\*Trams\*\*", "**Elétricos**"),
            (r"\*\*Trains\*\*", "**Comboios**"),
            (r"\*\*Metro\*\*", "**Metro**"),
            (r"\bLine\b", "Linha"),
            (r"\bBoard at\b", "Apanha em"),
            (r"\bExit at\b", "Sai em"),
            (r"\bTransfer at\b", "Transferência em"),
            (r"\bWalk from\b", "Caminha desde"),
            (r"\bWalk to\b", "Caminha até"),
            (r"\bReal time\b", "Tempo real"),
            (r"\bEstimated travel time\b", "Tempo estimado de viagem"),
            (r"\bNext departures\b", "Próximas partidas"),
            (r"\(stop\s+", "(paragem "),
            (r"\(Live\)", "(em tempo real)"),
            (r"\bLive\b", "Em tempo real"),
            (r"\bEm tempo real active\b", "tempo real ativo"),
            (r"\bReal time active\b", "tempo real ativo"),
            (r"\btempo\s+real ativo\b", "tempo real ativo"),
            (r"\blive active\b", "tempo real ativo"),
            (r"\bnormal service\b", "circulação normal"),
        ]

    normalized = text
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    if language == "pt":
        normalized = re.sub(r"\bpo+d+em\b", "podem", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\blistad+d?as\b", "listadas", normalized, flags=re.IGNORECASE)

    return normalized


def canonicalize_local_information_terms(text: str, language: str = "en") -> str:
    """Normalizes common PT-PT labels frequently leaked into EN local-information outputs."""
    if not text:
        return text

    if language == "en":
        replacements = [
            (r"\*\*Resumo da pesquisa\*\*", "**Search summary**"),
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
            (r"\*\*Categoria\*\*:", "**Category**:"),
            (r"\*\*Categoria:\*\*", "**Category:**"),
            (r"\*\*Quando\*\*:", "**When**:"),
            (r"\*\*Quando:\*\*", "**When:**"),
            (r"\*\*Dura(?:ç|c)[aã]o\*\*:", "**Duration**:"),
            (r"\*\*Dura(?:ç|c)[aã]o:\*\*", "**Duration:**"),
            (r"\*\*Bilhetes\*\*:", "**Buy tickets**:"),
            (r"\*\*Bilhetes:\*\*", "**Buy tickets:**"),
            (r"\*\*Local\*\*:", "**Location**:"),
            (r"\*\*Local:\*\*", "**Location:**"),
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
    else:
        replacements = [
            (r"\*\*Search summary\*\*", "**Resumo da pesquisa**"),
            (r"\*\*Brief description\*\*:", "**Breve descrição**:"),
            (r"\*\*Address\*\*:", "**Morada**:"),
            (r"\*\*Location\*\*:", "**Localização**:"),
            (r"\*\*Opening hours\*\*:", "**Horário**:"),
            (r"\*\*Quick tip\*\*:", "**Dica rápida**:"),
            (r"\*\*Tip\*\*:", "**Dica**:"),
            (r"\*\*Price\*\*:", "**Preço**:"),
            (r"\*\*Prices\*\*:", "**Preços**:"),
            (r"\*\*Buy tickets\*\*:", "**Comprar bilhetes**:"),
            (r"\*\*Official page\*\*", "**Site Oficial**"),
            (r"\*\*Category\*\*:", "**Categoria**:"),
            (r"\*\*Category:\*\*", "**Categoria:**"),
            (r"\*\*When\*\*:", "**Quando**:"),
            (r"\*\*When:\*\*", "**Quando:**"),
            (r"\*\*Duration\*\*:", "**Duração**:"),
            (r"\*\*Duration:\*\*", "**Duração:**"),
            (r"\*\*Local\*\*:", "**Local**:"),
            (r"\*\*Local:\*\*", "**Local:**"),
            (r"\bOpening hours:\s*check the official website\.?", "Horários de funcionamento: consultar website oficial."),
            (r"\bPrices:\s*check on site or on the official website\.?", "Preços: verificar no local ou website oficial."),
            (r"\bcheck on site or on the official website\b", "verificar no local ou website oficial"),
            (r"\bcheck the official website\b", "consultar website oficial"),
            (r"\bToday\b", "Hoje"),
            (r"\bClosed\b", "Fechado"),
            (r"\bNot specified\b", "Não especificado"),
            (r"\*\*Updated\*\*:", "**Atualizado**:"),
            (r"\*\*Updated:\*\*", "**Atualizado:**"),
            (r"\*\*Source\*\*:", "**Fonte**:"),
            (r"\*\*Source:\*\*", "**Fonte:**"),
        ]

    normalized = text
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    if language == "pt":
        normalized = localize_local_information_values(normalized, language=language)

    return normalized


def _translate_pt_category_value(value: str) -> str:
    """Translates common VisitLisboa category values into PT-PT without touching titles."""
    normalized_value = re.sub(r"\s+", " ", (value or "").strip().lower())
    return _PT_CATEGORY_VALUE_MAP.get(normalized_value, value.strip())


def _translate_pt_duration_value(value: str) -> str:
    """Translates common event duration values into PT-PT while preserving leading emojis."""
    raw_value = (value or "").strip()
    prefix_match = re.match(r"^([\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D\s]+)?(.+?)$", raw_value)
    if not prefix_match:
        return raw_value

    prefix = (prefix_match.group(1) or "").strip()
    core = (prefix_match.group(2) or "").strip()
    mapped = _PT_DURATION_VALUE_MAP.get(re.sub(r"\s+", " ", core.lower()), core)
    if prefix:
        return f"{prefix} {mapped}".strip()
    return mapped


def localize_local_information_values(text: str, language: str = "en") -> str:
    """Localizes common metadata values that remain in English inside PT-PT researcher/planner outputs."""
    if not text or language != "pt":
        return text

    localized_lines: list[str] = []
    for line in text.splitlines():
        updated_line = line

        updated_line = re.sub(
            r"(\*\*(?:Categoria|Category)\*\*:\s*)(.+)$",
            lambda match: f"{match.group(1)}{_translate_pt_category_value(match.group(2))}",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"(\*\*(?:Categoria|Category):\*\*\s*)(.+)$",
            lambda match: f"{match.group(1)}{_translate_pt_category_value(match.group(2))}",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"\*\*Description\*\*:",
            "**Descrição**:",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"\*\*Description:\*\*",
            "**Descrição:**",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"\*\*Filter used\*\*:",
            "**Filtro aplicado**:",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"\*\*Filter used:\*\*",
            "**Filtro aplicado:**",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"\*\*Result count\*\*:",
            "**Resultado do filtro**:",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"\*\*Result count:\*\*",
            "**Resultado do filtro:**",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"\*\*Highlights shown\*\*:",
            "**Destaques mostrados**:",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"\*\*Highlights shown:\*\*",
            "**Destaques mostrados:**",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"(\*\*(?:Duração|Duration)\*\*:\s*|\*\*(?:Duração|Duration):\*\*\s*)(.+)$",
            lambda match: f"{match.group(1)}{_translate_pt_duration_value(match.group(2))}",
            updated_line,
            flags=re.IGNORECASE,
        )

        if "Preço" in updated_line or "Price" in updated_line:
            updated_line = re.sub(
                r"\bFrom\s+(€?\d+(?:[\.,]\d+)?)\s+to\s+(€?\d+(?:[\.,]\d+)?)\b",
                r"de \1 a \2",
                updated_line,
                flags=re.IGNORECASE,
            )
            updated_line = re.sub(
                r"\bFrom\s+(€?\d+(?:[\.,]\d+)?)\b",
                r"desde \1",
                updated_line,
                flags=re.IGNORECASE,
            )
            updated_line = re.sub(r"\bFree\b", "Gratuito", updated_line, flags=re.IGNORECASE)
            updated_line = re.sub(r"\bOn request\b", "Sob consulta", updated_line, flags=re.IGNORECASE)
            updated_line = re.sub(r"\bSold out\b", "Esgotado", updated_line, flags=re.IGNORECASE)

        if "TripAdvisor" in updated_line:
            updated_line = re.sub(
                r"\(([0-9\.,]+)\s+reviews?\)",
                r"(\1 avaliações)",
                updated_line,
                flags=re.IGNORECASE,
            )

        if "Quando" in updated_line or "When" in updated_line:
            updated_line = re.sub(r"\bat\s+(\d{1,2}:\d{2})\b", r"às \1", updated_line)

        if "📍" in updated_line or "**Local" in updated_line or "**Location" in updated_line:
            updated_line = re.sub(r"\bLisbon\b", "Lisboa", updated_line)

        # Handle english days and schedules returned by the APIs directly
        updated_line = re.sub(r"\bToday:", "Hoje:", updated_line, flags=re.IGNORECASE)
        updated_line = re.sub(r"\bTomorrow:", "Amanhã:", updated_line, flags=re.IGNORECASE)
        updated_line = re.sub(r"\bMonday:", "Segunda-feira:", updated_line, flags=re.IGNORECASE)
        updated_line = re.sub(r"\bTuesday:", "Terça-feira:", updated_line, flags=re.IGNORECASE)
        updated_line = re.sub(r"\bWednesday:", "Quarta-feira:", updated_line, flags=re.IGNORECASE)
        updated_line = re.sub(r"\bThursday:", "Quinta-feira:", updated_line, flags=re.IGNORECASE)
        updated_line = re.sub(r"\bFriday:", "Sexta-feira:", updated_line, flags=re.IGNORECASE)
        updated_line = re.sub(r"\bSaturday:", "Sábado:", updated_line, flags=re.IGNORECASE)
        updated_line = re.sub(r"\bSunday:", "Domingo:", updated_line, flags=re.IGNORECASE)

        # Handle English labels from VisitLisboa / Researcher outputs by targeting words directly
        # preserving entirely the surrounding bold (**), emojis, spaces, or colons used by the LLM
        updated_line = re.sub(r"(?i)\bBrief\s+description\b", "Descrição", updated_line)
        updated_line = re.sub(r"(?i)\bDescription\b", "Descrição", updated_line)
        updated_line = re.sub(r"(?i)\bAddress\b", "Morada", updated_line)
        updated_line = re.sub(r"(?i)\bLocation\b", "Localização", updated_line)
        updated_line = re.sub(r"(?i)\bOpening\s+hours\b", "Horário", updated_line)
        updated_line = re.sub(r"(?i)\bSchedule\b", "Horário", updated_line)
        updated_line = re.sub(r"(?i)\bTip\b", "Dica", updated_line)
        updated_line = re.sub(r"(?i)\bPrice\b", "Preço", updated_line)
        updated_line = re.sub(r"(?i)\bAccessibility\b", "Acessibilidade", updated_line)
        updated_line = re.sub(r"(?i)\bParking\b", "Estacionamento", updated_line)
        updated_line = re.sub(r"(?i)\bPublic\s+transport\s+access\b", "Acessos por transportes públicos", updated_line)
        updated_line = re.sub(r"(?i)\bContact\b", "Contacto", updated_line)
        updated_line = re.sub(r"(?i)\bTemporary\s+requirements\b", "Exigências temporárias", updated_line)
        updated_line = re.sub(r"(?i)\bReservations\b", "reservas", updated_line)
        updated_line = re.sub(r"(?i)\bEducational\s+programs\b", "Programas educativos", updated_line)
        updated_line = re.sub(r"(?i)\bGuided\s+tours\b", "visitas guiadas", updated_line)

        updated_line = re.sub(
            r"\*\*Total matching events:\*\*",
            "**Total de eventos encontrados:**",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"\*\*Source completeness note:\*\*",
            "**Nota sobre a completude da fonte:**",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"([0-9]+)\s+matching event\(s\) in VisitLisboa do not include confirmed dates, so they were excluded from the '([^']+)' date window\.",
            r"\1 evento(s) no VisitLisboa não incluem datas confirmadas, por isso foram excluídos da janela temporal '\2'.",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"([0-9]+)\s+additional matching record\(s\) were excluded because the source does not confirm their dates yet\.",
            r"\1 registo(s) adicional(is) compatíveis foram excluídos porque a fonte ainda não confirma a respetiva data.",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"([0-9]+)\s+confirmed-date event\(s\) match this filter\.",
            r"\1 evento(s) com data confirmada correspondem a este filtro.",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"([0-9]+)\s+most relevant result\(s\)\.",
            r"\1 resultado(s) mais relevantes.",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"([0-9]+)\s+most relevant result\(s\)\s*\(window\s+([0-9]+-[0-9]+)\)\.",
            r"\1 resultado(s) mais relevantes (janela \2).",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"\ball categories\b",
            "todas as categorias",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(r"\bthis week\b", "esta semana", updated_line, flags=re.IGNORECASE)
        updated_line = re.sub(r"\bthis weekend\b", "este fim de semana", updated_line, flags=re.IGNORECASE)
        updated_line = re.sub(r"\bnext week\b", "próxima semana", updated_line, flags=re.IGNORECASE)
        updated_line = re.sub(r"\bthis month\b", "este mês", updated_line, flags=re.IGNORECASE)
        updated_line = re.sub(r"\bnext month\b", "próximo mês", updated_line, flags=re.IGNORECASE)
        updated_line = re.sub(
            r"(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})",
            r"\1 a \2",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"\bbroad event discovery\b",
            "pesquisa geral de eventos",
            updated_line,
            flags=re.IGNORECASE,
        )
        updated_line = re.sub(
            r"\(date not confirmed in source\)",
            "(data não confirmada na fonte)",
            updated_line,
            flags=re.IGNORECASE,
        )

        localized_lines.append(updated_line)

    return "\n".join(localized_lines)


def strip_technical_output_artifacts(text: str) -> str:
    """Removes backend-oriented metadata that should not appear in final user answers."""
    if not text:
        return text

    cleaned_lines: list[str] = []
    technical_patterns = [
        re.compile(r"^\s*(?:[-*•]\s*)?(?:🗺️\s*)?GPS\s*:", re.IGNORECASE),
        re.compile(r"^\s*(?:[-*•]\s*)?(?:🚏\s*)?(?:next\s+)?stop(?:_id|\s+id)\s*[:=]", re.IGNORECASE),
        re.compile(r"^\s*(?:[-*•]\s*)?(?:🚏\s*)?(?:\*\*(?:next\s+)?stop(?:_id|\s+id)\*\*)\s*[:=]", re.IGNORECASE),
        re.compile(r"^\s*(?:[-*•]\s*)?(?:line|route|pattern|trip)(?:_id|\s+id)\s*[:=]", re.IGNORECASE),
        re.compile(r"^\s*(?:[-*•]\s*)?(?:\*\*(?:Vehicle|Ve[ií]culo)\*\*|(?:Vehicle|Ve[ií]culo))\s*:", re.IGNORECASE),
        re.compile(r"^\s*(?:[-*•]\s*)?(?:\*\*(?:Plate|Matrícula|Matricula)\*\*|(?:Plate|Matrícula|Matricula))\s*:", re.IGNORECASE),
    ]
    placeholder_line = re.compile(
        r"\b(?:Unknown event|Evento sem nome|Unknown place|Local sem nome|Unknown station|Estação sem nome)\b",
        re.IGNORECASE,
    )
    empty_value_line = re.compile(
        r"^\s*(?:[-*•]\s*)?(?:[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]\s*)?(?:\*\*[^*]+\*\*\s*:?\s*)?(?:N/?A|Unknown|UNKNOWN|Não disponível|Nao disponivel|Not available)\s*$",
        re.IGNORECASE,
    )

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if any(pattern.match(stripped) for pattern in technical_patterns):
            continue
        if placeholder_line.search(stripped):
            continue
        if empty_value_line.match(stripped):
            continue
        cleaned_lines.append(raw_line)

    cleaned = "\n".join(cleaned_lines).strip()
    inline_replacements = [
        (r"\s*-\s*(?:📍\s*)?GPS\s*:\s*[^\n]+?(?=(?:\s+-\s+|\s+📌|$))", ""),
        (r"\s*-\s*(?:🚏\s*)?(?:next\s+)?stop(?:_id|\s+id)\s*[:=]\s*[^\n]+?(?=(?:\s+-\s+|\s+📌|$))", ""),
        (r"\s*-\s*(?:\*\*(?:Plate|Matrícula|Matricula)\*\*|(?:Plate|Matrícula|Matricula))\s*:\s*[^\n]+?(?=(?:\s+-\s+|\s+📌|$))", ""),
        (r"GPS\s*:\s*\**-?\d{1,2}\.\d+\**\s*,\s*\**-?\d{1,3}\.\d+\**", ""),
        (r"\*\*GPS\*\*\s*:\s*\**-?\d{1,2}\.\d+\**\s*,\s*\**-?\d{1,3}\.\d+\**", ""),
        (r"(?:\|\s*)?ve[ií]culo\s*:\s*\**[A-Za-z0-9_-]+\**(?:\s*\(m[áa]tr[ií]cula\s*\**[A-Za-z0-9-]+\**\))?", ""),
        (r"(?:\|\s*)?\*\*Ve[ií]culo\*\*\s*:\s*\**[A-Za-z0-9_-]+\**(?:\s*\(Matr[íi]c\w*\s*\**[A-Za-z0-9-]+\**\))?", ""),
        (r"(?:\|\s*)?Matr[íi]c\w*\s*:\s*\**[A-Za-z0-9-]+\**", ""),
        (r"\*\*Ve[ií]culo\s+\**[A-Za-z0-9_-]+\**(?:\s*\(m[áa]tr[ií]c\w*\s*\**[A-Za-z0-9-]+\**\))?\*\*", ""),
        (r"Ve[ií]culo\s+\**[A-Za-z0-9_-]+\**(?:\s*\(m[áa]tr[ií]c\w*\s*\**[A-Za-z0-9-]+\**\))?", ""),
        (r"(?:\|\s*)?vehicle\s*:\s*\**[A-Za-z0-9_-]+\**(?:\s*\(plate\s*\**[A-Za-z0-9-]+\**\))?", ""),
        (r"(?:\|\s*)?\*\*Vehicle\*\*\s*:\s*\**[A-Za-z0-9_-]+\**(?:\s*\(Plate\s*\**[A-Za-z0-9-]+\**\))?", ""),
        (r"\*\*Vehicle\s+\**[A-Za-z0-9_-]+\**(?:\s*\(plate\s*\**[A-Za-z0-9-]+\**\))?\*\*", ""),
        (r"Vehicle\s+\**[A-Za-z0-9_-]+\**(?:\s*\(plate\s*\**[A-Za-z0-9-]+\**\))?", ""),
        (r"\s*\((?:paragem|stop)\s+id\s*[:#]?\s*\**\d+\**\)", ""),
        (r"\s*[—-]?\s*(?:paragem|stop)\s+id\s*[:#]?\s*\**\d+\**", ""),
        (r"\bID\s*:\s*\**\d+\**", ""),
        (r"\s*\((?:id)\s*[:#]?\s*\d+\)", ""),
        (r"[;,]\s*viatura\s+\**[A-Za-z0-9_-]+\**(?:\s*,\s*m[áa]tr[ií]cula\s+\**[A-Za-z0-9_-]+\**)?", ""),
        (r"[;,]\s*vehicle\s+\**[A-Za-z0-9_-]+\**(?:\s*,\s*plate\s+\**[A-Za-z0-9_-]+\**)?", ""),
        (r"\(\s*vehicle\s+[A-Za-z0-9_-]+\s*,\s*([^)]+)\)", r"(\1)"),
        (r"\(\s*ve[ií]culo\s+[A-Za-z0-9_-]+\s*,\s*([^)]+)\)", r"(\1)"),
        (r"\(\s*vehicle\s+[A-Za-z0-9_-]+\s*\)", ""),
        (r"\(\s*ve[ií]culo\s+[A-Za-z0-9_-]+\s*\)", ""),
    ]
    for pattern, replacement in inline_replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\bHorario\b", "Horário", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(\s*([^()]*?)\s*[;,]\s*\)", r"(\1)", cleaned)
    cleaned = re.sub(r";\s*;", ";", cleaned)
    cleaned = re.sub(r"\(\s*,\s*", "(", cleaned)
    cleaned = re.sub(r"\(\s*;\s*", "(", cleaned)
    cleaned = re.sub(r"(?i)valide\s+a\s+entrada\s+na\s+paragem\s+com\s+o\s+id\s+se\s+estiver\s+noutro\s+abrigo\.?", "", cleaned)
    cleaned = re.sub(r"\|\s*\|", "|", cleaned)
    cleaned = re.sub(r"—\s*—", "—", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n\s+\n", "\n\n", cleaned)
    return cleaned.strip()


def sanitize_event_title_suffixes(text: str) -> str:
    """Drops slug-like numeric suffixes from event titles when they leak into the UI."""
    if not text:
        return text

    updated_lines: list[str] = []
    title_pattern = re.compile(
        r"^(\s*(?:[-*•]\s*)?(?:\d+\.\s*)?📅\s+\*\*[^*]+?)\s+(0\d{2,3}|\d{2,4})(\*\*)(.*)$"
    )

    for raw_line in text.splitlines():
        match = title_pattern.match(raw_line)
        if match:
            raw_title = re.sub(r"\*\*", "", match.group(1)).strip()
            title_word_count = len(raw_title.split())
            if title_word_count >= 2:
                raw_line = f"{match.group(1)}{match.group(3)}{match.group(4)}"
        updated_lines.append(raw_line)

    return "\n".join(updated_lines)


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


def strip_researcher_meta_notes(text: str) -> str:
    """Remove researcher-side QA/meta caveats that should not leak to users."""
    if not text:
        return text

    meta_patterns = [
        re.compile(
            r"^\s*(?:[-*•]\s*)?⚠️\s+Alguns eventos não indicam preço\..*domínios conhecidos.*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*•]\s*)?⚠️\s+Some events do not list a price\..*known domains.*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*•]\s*)?⚠️\s+Os URLs apresentados parecem usar domínios conhecidos.*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*•]\s*)?⚠️\s+The URLs shown appear to use known domains.*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*•]\s*)?⚠️\s+Alguns eventos repetem-se em várias datas.*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*•]\s*)?⚠️\s+A disponibilidade, datas, horários e preços devem ser confirmados.*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*•]\s*)?⚠️\s+Os links podem variar entre versões.*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*•]\s*)?⚠️\s+Alguns eventos não apresentam hora exata e/ou preço indicado na fonte\.?$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*•]\s*)?⚠️\s+Há mistura de idioma nos links/URLs .*campos principais estão em português\.?$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*•]\s*)?⚠️\s+As datas e preços acima devem ser confirmados no VisitLisboa.*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*•]\s*)?⚠️\s+Em alguns eventos, o preço não está disponível nos dados.*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*•]\s*)?⚠️\s+Alguns eventos usam datas amplas ou múltiplas ocorrências.*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*•]\s*)?⚠️\s+Some events repeat across multiple dates.*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*•]\s*)?⚠️\s+Availability, dates, times, and prices should be confirmed.*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*•]\s*)?⚠️\s+Links may vary across versions.*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*•]\s*)?⚠️\s+Some events do not show an exact time and/or price in the source\.?$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*•]\s*)?⚠️\s+There is mixed language in the links/URLs .*main fields remain in Portuguese\.?$",
            re.IGNORECASE,
        ),
    ]

    kept_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(pattern.match(stripped) for pattern in meta_patterns):
            continue
        kept_lines.append(line)

    return clean_newlines("\n".join(kept_lines)).strip()


def _infer_service_heading_from_dataset(
    dataset_title: str,
    language: str = "en",
) -> tuple[str, str]:
    """Infer a stable section heading and item icon for nearby-service datasets."""
    normalized_title = (dataset_title or "").strip()
    lowered_title = normalized_title.lower()
    near_match = re.search(r"\((?:near|perto de)\s+(.+?)\)$", normalized_title, re.IGNORECASE)
    near_location = near_match.group(1).strip() if near_match else ""

    service_catalog = [
        ("💊", "Farmácias", "Pharmacies", ("farm", "parafarm")),
        ("🏥", "Hospitais", "Hospitals", ("hospital",)),
        ("📚", "Bibliotecas", "Libraries", ("bibliot", "library")),
        ("🎓", "Escolas", "Schools", ("escola", "school")),
        ("🌿", "Jardins", "Gardens", ("jardim", "garden", "park", "parque")),
        ("👮", "Polícia", "Police", ("polic",)),
    ]

    for icon, pt_label, en_label, markers in service_catalog:
        if any(marker in lowered_title for marker in markers):
            label = pt_label if language == "pt" else en_label
            if near_location:
                if language == "pt":
                    return f"#### {icon} {label} perto de {near_location}", icon
                return f"#### {icon} {label} near {near_location}", icon
            return f"#### {icon} {label}", icon

    generic_label = "Serviços próximos" if language == "pt" else "Nearby services"
    if near_location:
        if language == "pt":
            return f"#### 📍 {generic_label} perto de {near_location}", "📍"
        return f"#### 📍 {generic_label} near {near_location}", "📍"
    return f"#### 📍 {generic_label}", "📍"


def structure_ranked_research_results(text: str) -> str:
    """Converts flat numbered researcher results into nested markdown lists."""
    if not text:
        return text

    entry_re = re.compile(r"^\s*(\d+)\.\s+(.+)$")
    summary_re = re.compile(r"^(?:📊|⚠️|💡|📌|###|##|#)")
    structured_lines: list[str] = []
    inside_ranked_item = False
    child_prefix = "    - "

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        if not stripped:
            if structured_lines and structured_lines[-1] != "":
                structured_lines.append("")
            inside_ranked_item = False
            continue

        entry_match = entry_re.match(stripped)
        if entry_match:
            structured_lines.append(f"- {entry_match.group(2).strip()}")
            inside_ranked_item = True
            continue

        if inside_ranked_item:
            if summary_re.match(stripped):
                if structured_lines and structured_lines[-1] != "":
                    structured_lines.append("")
                structured_lines.append(stripped)
                structured_lines.append("")
                inside_ranked_item = False
                continue

            if stripped.startswith(("- ", "* ", "• ")):
                normalized_bullet = stripped.replace("• ", "- ", 1)
                structured_lines.append(f"    {normalized_bullet}")
                continue

            structured_lines.append(f"{child_prefix}{stripped}")
            continue

        if summary_re.match(stripped):
            if structured_lines and structured_lines[-1] != "":
                structured_lines.append("")
            structured_lines.append(stripped)
            structured_lines.append("")
            continue

        if raw_line[:1].isspace() and stripped.startswith(("- ", "* ", "• ")):
            structured_lines.append(raw_line.rstrip())
            continue

        structured_lines.append(stripped)

    return clean_newlines("\n".join(structured_lines)).strip()


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


# ==========================================================================
# Transport source-line operator filter (Phase 1.4)
# ==========================================================================

_OPERATOR_SOURCE_LINKS: Dict[str, str] = {
    "metro": "[*Metro de Lisboa*](https://www.metrolisboa.pt)",
    "carris": "[*Carris*](https://www.carris.pt)",
    "carris_metropolitana": "[*Carris Metropolitana*](https://www.carrismetropolitana.pt)",
    "cp": "[*CP*](https://www.cp.pt)",
}


def operators_from_tool_names(tool_names) -> List[str]:
    """Map a list of invoked tool names to the set of transport operators used.

    The order of operators in the returned list reflects the canonical display
    order: metro, carris urban, carris metropolitana, CP. Tool-name grouping
    matches the exports in ``tools/__init__.py``:

    - Metro de Lisboa: ``get_metro_*``, ``find_nearest_metro``, ``get_all_metro_stations``.
    - Carris Urban: ``carris_*`` (but NOT ``carris_metropolitana_*``).
    - Carris Metropolitana: ``*carris_metropolitana*``, plus the bus family
      (``find_bus_routes``, ``find_direct_bus_lines``, ``get_bus_*``,
      ``get_real_time_bus_positions``).
    - CP: ``get_train_*``, ``search_cp_stations``, ``get_cp_routes``, ``plan_train_trip``.
        - ``get_transport_summary`` is a true multi-operator overview and must cite
            Metro, Carris, Carris Metropolitana, and CP.
        - ``get_route_between_stations`` keeps the source line produced by the tool
            itself; we do not guess extra operators from the wrapper alone.
    """
    if not tool_names:
        return []
    invoked = set()
    for name in tool_names:
        low = str(name or "").lower()
        if not low:
            continue
        # Carris Metropolitana (check BEFORE generic carris_ prefix)
        if "carris_metropolitana" in low:
            invoked.add("carris_metropolitana")
            continue
        if low in {
            "find_bus_routes",
            "find_direct_bus_lines",
            "get_bus_next_departures",
            "get_bus_realtime_locations",
            "get_real_time_bus_positions",
        }:
            invoked.add("carris_metropolitana")
            continue
        # Carris Urban
        if low.startswith("carris_"):
            invoked.add("carris")
            continue
        # CP
        if low.startswith("get_train_") or low in {
            "search_cp_stations",
            "get_cp_routes",
            "plan_train_trip",
        }:
            invoked.add("cp")
            continue
        # Metro
        if low.startswith("get_metro_") or low in {
            "find_nearest_metro",
            "get_all_metro_stations",
        }:
            invoked.add("metro")
            continue
        if low == "get_transport_summary":
            invoked.update({"metro", "carris", "carris_metropolitana", "cp"})
            continue
        if low == "get_route_between_stations":
            continue
    order = ["metro", "carris", "carris_metropolitana", "cp"]
    return [op for op in order if op in invoked]


def rebuild_transport_source_line(
    text: str,
    operators_used: List[str],
    language: str = "en",
) -> str:
    """Rewrite the final 📌 Source/Fonte footer to list only the operators actually used.

    Fixes the Q2 bug where a Carris-only answer still cites CP because the prompt
    template lists all networks. If ``operators_used`` is empty, the original
    text is returned unchanged (we do not invent sources).
    """
    if not text or not isinstance(text, str):
        return text or ""
    if not operators_used:
        return text

    deduped = []
    for op in operators_used:
        link = _OPERATOR_SOURCE_LINKS.get(op)
        if link and link not in deduped:
            deduped.append(link)
    if not deduped:
        return text

    timestamp = extract_update_time(text) or datetime.now().strftime("%H:%M")
    if language == "pt":
        replacement = f"📌 **Fonte:** {' | '.join(deduped)} | **Atualizado:** {timestamp}"
    else:
        replacement = f"📌 **Source:** {' | '.join(deduped)} | **Updated:** {timestamp}"

    # Only replace an EXISTING transport-style source line, do not append.
    # That preserves researcher / weather / planner sources on multi-agent runs.
    def _is_transport_source(line: str) -> bool:
        stripped = line.strip()
        if not _SOURCE_LINE_RE.match(stripped):
            return False
        low = stripped.lower()
        transport_markers = (
            "metro",
            "metrolisboa",
            "carris",
            "cp",
            "cp.pt",
            "carrismetropolitana",
            "metro de lisboa",
        )
        return any(marker in low for marker in transport_markers)

    return _replace_source_line(text, replacement, predicate=_is_transport_source)


def _looks_like_mojibake(text: str) -> bool:
    """Best-effort detector for common UTF-8/Windows-1252 mojibake fragments."""
    return any(fragment in (text or "") for fragment in ("Ã", "Â", "ðŸ", "\x8d", "\x8f"))


def _strip_accents_compat(value: str) -> str:
    """Accent-insensitive normalization helper used by robust formatters."""
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def structure_service_lookup_markdown(text: str, language: str = "en") -> str:
    """Convert nearby-service dumps into stable markdown, including mojibake inputs."""
    if not text or "results from '" not in text.lower():
        return text

    is_pt = language == "pt"
    mojibake = _looks_like_mojibake(text)
    header_re = re.compile(r"Found\s+\d+\s+results?\s+from\s+'(?P<title>[^']+)':", re.IGNORECASE)
    item_re = re.compile(r"^(?:\*\*)?(?P<num>\d+)\.?(?:\*\*)?\s+(?P<name>.+?)\s*$")

    address_label = "Morada" if is_pt else "Address"
    distance_label = "DistÃ¢ncia" if (is_pt and mojibake) else ("Distância" if is_pt else "Distance")
    coords_label = "Coordenadas" if is_pt else "Coordinates"
    source_line = (
        f"\U0001F4CC **Fonte:** [*Lisboa Aberta*](https://dados.cm-lisboa.pt/) | **Atualizado:** {datetime.now().strftime('%H:%M')}"
        if is_pt
        else f"\U0001F4CC **Source:** [*Lisboa Aberta*](https://dados.cm-lisboa.pt/) | **Updated:** {datetime.now().strftime('%H:%M')}"
    )

    def normalize_entry_value(raw_value: str) -> str:
        cleaned_value = _strip_leading_section_emoji(
            _strip_markdown_formatting(raw_value).strip()
        )
        if is_pt:
            distance_match = re.search(r"(?P<km>\d+(?:\.\d+)?)\s*km\b", cleaned_value, re.IGNORECASE)
            if distance_match:
                cleaned_value = f"{distance_match.group('km')} km"
        return cleaned_value.strip()

    def infer_heading(dataset_title: str) -> tuple[str, str]:
        normalized_title = _strip_accents_compat(dataset_title).lower()
        raw_location_match = re.search(r"\((?:near|perto de)\s+([^)]+)\)", dataset_title, re.IGNORECASE)
        location = raw_location_match.group(1).strip() if raw_location_match else ""

        if "farm" in normalized_title:
            heading = (
                f"Farmácias Perto de {location}" if is_pt and location else
                "Farmácias Próximas" if is_pt else
                f"Pharmacies Near {location}" if location else
                "Nearby Pharmacies"
            )
            return f"#### \U0001F48A {heading}", "\U0001F48A"
        if "hospit" in normalized_title:
            is_public_hospital = any(marker in normalized_title for marker in ("public", "publico", "publicos", "publica", "publicas"))
            heading = (
                f"Hospitais Públicos Perto de {location}" if is_pt and is_public_hospital and location else
                f"Hospitais Perto de {location}" if is_pt and location else
                "Hospitais Públicos Próximos" if is_pt and is_public_hospital else
                "Hospitais Próximos" if is_pt else
                f"Public Hospitals Near {location}" if is_public_hospital and location else
                f"Hospitals Near {location}" if location else
                "Nearby Public Hospitals" if is_public_hospital else
                "Nearby Hospitals"
            )
            return f"#### \U0001F3E5 {heading}", "\U0001F3E5"
        if "polic" in normalized_title:
            heading = (
                f"Polícia Perto de {location}" if is_pt and location else
                "Polícia Próxima" if is_pt else
                f"Police Near {location}" if location else
                "Nearby Police"
            )
            return f"#### \U0001F46E {heading}", "\U0001F46E"
        return f"#### \U0001F4CD {dataset_title.strip()}", "\U0001F4CD"

    lines = text.splitlines()
    structured_lines: list[str] = []
    pending_heading: Optional[str] = None
    transformed = False
    index = 0

    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue

        if re.match(r"^#{3,4}\s+", stripped):
            pending_heading = stripped
            index += 1
            continue

        normalized_header = _strip_accents_compat(_strip_markdown_formatting(stripped))
        header_match = header_re.search(normalized_header)
        if not header_match:
            if pending_heading:
                structured_lines.extend([pending_heading, ""])
                pending_heading = None
            structured_lines.append(stripped)
            index += 1
            continue

        transformed = True
        dataset_title = header_match.group("title").strip()
        auto_heading, item_icon = infer_heading(dataset_title)
        section_heading = pending_heading or auto_heading
        pending_heading = None
        structured_lines.extend([section_heading, ""])
        index += 1

        entries: list[dict[str, str]] = []
        current_entry: Optional[dict[str, str]] = None
        while index < len(lines):
            current_line = lines[index].strip()
            if not current_line:
                index += 1
                continue

            normalized_current = _strip_accents_compat(_strip_markdown_formatting(current_line))
            if re.match(r"^#{3,4}\s+", current_line) or header_re.search(normalized_current):
                break

            plain_line = _strip_markdown_formatting(current_line).strip()
            item_match = item_re.match(plain_line)
            if item_match:
                current_entry = {
                    "name": normalize_entry_value(item_match.group("name")),
                    "address": "",
                    "distance": "",
                    "coords": "",
                }
                entries.append(current_entry)
                index += 1
                continue

            if current_entry is not None:
                normalized_value = normalize_entry_value(plain_line)
                lowered_plain = _strip_accents_compat(normalized_value).lower()
                if re.search(r"\(-?\d+\.\d+,\s*-?\d+\.\d+\)", normalized_value):
                    current_entry["coords"] = normalized_value
                elif "km away" in lowered_plain or re.search(r"\b\d+(?:\.\d+)?\s*km\b", lowered_plain):
                    current_entry["distance"] = normalized_value
                elif not current_entry["address"]:
                    current_entry["address"] = normalized_value

            index += 1

        for item_number, entry in enumerate(entries, 1):
            structured_lines.append(f"{item_number}. {item_icon} **{entry['name']}**")
            if entry["address"]:
                structured_lines.append(f"   \U0001F4CD **{address_label}:** {entry['address']}")
            if entry["distance"]:
                structured_lines.append(f"   \U0001F4CF **{distance_label}:** {entry['distance']}")
            if entry["coords"]:
                structured_lines.append(f"   \U0001F5FA\uFE0F **{coords_label}:** {entry['coords']}")
            structured_lines.append("")

    if pending_heading:
        structured_lines.extend([pending_heading, ""])

    structured = clean_newlines("\n".join(structured_lines)).strip()
    if not transformed:
        return text
    if structured and not has_source_line(structured):
        structured = f"{structured}\n\n{source_line}".strip()
    return structured


_RESEARCHER_CARD_START_RE = re.compile(
    r"^(?:[-*]\s+|\d+\.\s+)(?![📂📍🕐⭐📞🔗🌐💶🎟️📝])(?P<emoji>\S+)\s+\*\*(?P<title>.+?)\*\*\s*$"
)


def _researcher_card_labels(language: str) -> dict[str, str]:
    """Return localized field labels for canonical researcher cards."""
    if language == "pt":
        return {
            "description": "Descrição",
            "category": "Categoria",
            "address": "Morada",
            "phone": "Telefone",
            "rating": "Avaliação",
            "price": "Preço",
            "website": "Website",
            "tickets": "Bilhetes",
            "today": "Hoje",
            "hours": "Horário",
            "distance": "Distância",
            "coordinates": "Coordenadas",
        }
    return {
        "description": "Description",
        "category": "Category",
        "address": "Address",
        "phone": "Phone",
        "rating": "Rating",
        "price": "Price",
        "website": "Website",
        "tickets": "Tickets",
        "today": "Today",
        "hours": "Hours",
        "distance": "Distance",
        "coordinates": "Coordinates",
    }


def _render_researcher_link_value(value: str, label: str) -> str:
    """Render website or ticket values as markdown links when possible."""
    stripped = (value or "").strip()
    if not stripped:
        return stripped
    if "](" in stripped:
        return stripped
    url_match = re.search(r"https?://\S+", stripped)
    if not url_match:
        return stripped
    url = url_match.group(0).rstrip(").,;")
    parsed = urlparse(url)
    netloc = (parsed.netloc or url).replace("www.", "")
    if label.lower() in {"tickets", "bilhetes"}:
        return f"[{label}]({url})"
    return f"[{netloc}]({url})"


def _extract_first_url(value: str) -> str:
    """Return the first URL found in a string, trimmed of trailing punctuation."""
    match = re.search(r"https?://\S+", value or "")
    return match.group(0).rstrip(").,;") if match else ""


_CANONICAL_PLACE_CARD_START_RE = re.compile(r"^###\s+(?P<emoji>\S+)\s+(?P<title>.+?)\s*$")


def _iter_structured_place_card_sections(text: str) -> List[List[str]]:
    """Split canonical or pre-canonical place-card markdown into per-card sections."""
    sections: List[List[str]] = []
    current_section: List[str] = []

    for raw_line in (text or "").splitlines():
        stripped = raw_line.strip()
        if _CANONICAL_PLACE_CARD_START_RE.match(stripped) or _RESEARCHER_CARD_START_RE.match(stripped):
            if current_section:
                sections.append(current_section)
            current_section = [raw_line]
            continue
        if current_section:
            current_section.append(raw_line)

    if current_section:
        sections.append(current_section)

    return sections


def _count_structured_place_cards(text: str) -> int:
    """Count place cards that still preserve the canonical structured layout."""
    return len(_iter_structured_place_card_sections(text))


def _place_response_missing_required_fields(
    text: str,
    expected_language: str,
    place_card_count: int,
) -> bool:
    """Return whether any structured place card is missing canonical core fields."""
    if place_card_count <= 0:
        return True

    sections = _iter_structured_place_card_sections(text)
    if len(sections) < place_card_count:
        return True

    for section in sections[:place_card_count]:
        section_text = "\n".join(section)
        normalized = _strip_accents_compat(_strip_markdown_formatting(section_text)).lower()

        has_description = bool(
            re.search(r"\b(description|descricao)\b", normalized)
            or re.search(r"^\s*[-*]\s+📝", section_text, re.MULTILINE)
        )
        has_address = bool(
            re.search(r"\b(address|morada|location|localizacao|endereco)\b", normalized)
            or re.search(r"^\s*[-*]\s+📍", section_text, re.MULTILINE)
        )
        has_hours = bool(
            re.search(r"\b(hours|opening hours|today|horario|horarios de funcionamento|hoje)\b", normalized)
            or re.search(r"^\s*[-*]\s+🕐", section_text, re.MULTILINE)
            or re.search(r"\b(check the official website|consultar website oficial)\b", normalized)
        )
        has_website = bool(
            re.search(r"\b(website|site oficial|official page|url)\b", normalized)
            or "http://" in normalized
            or "https://" in normalized
            or re.search(r"\b(check the official website|consultar website oficial)\b", normalized)
        )

        if not (has_description and has_address and has_hours and has_website):
            return True

    return False


def _is_researcher_event_meta_line(text: str) -> bool:
    """Return whether a line is a search-summary or generic follow-up note for event lists."""
    normalized = _strip_accents_compat(_strip_markdown_formatting(text or "")).lower().strip()
    meta_prefixes = (
        "resumo da pesquisa",
        "search summary",
        "filtro aplicado",
        "filter used",
        "resultado do filtro",
        "result count",
        "destaques mostrados",
        "highlights shown",
        "a lista mostra",
        "this list shows",
        "nota sobre a completude da fonte",
        "source completeness note",
    )
    if any(normalized.startswith(prefix) for prefix in meta_prefixes):
        return True
    return normalized.startswith(("🧾 ", "🧭 ", "📊 ", "✨ ", "💡 "))


def _is_specific_lookup_fallback_intro(text: str) -> bool:
    """Return whether a line explicitly says the requested named item was not found exactly."""
    normalized = _strip_accents_compat(_strip_markdown_formatting(text or "")).lower().strip()
    patterns = (
        "nao encontrei um evento especifico com o nome",
        "nao encontrei um local especifico com o nome",
        "i could not find a specific event named",
        "i could not find a specific place named",
    )
    return any(pattern in normalized for pattern in patterns)


def _select_researcher_specific_lookup_intro(
    primary_intro: list[str],
    fallback_intro: list[str],
) -> list[str]:
    """Prefer grounded exact-not-found intros over generic QA-rewritten intros."""
    explicit_primary = [line for line in primary_intro if _is_specific_lookup_fallback_intro(line)]
    if explicit_primary:
        return primary_intro

    explicit_fallback = [line for line in fallback_intro if _is_specific_lookup_fallback_intro(line)]
    if not explicit_fallback:
        return []

    heading_lines = [line for line in primary_intro if line.strip().startswith("### ")] or [
        line for line in fallback_intro if line.strip().startswith("### ")
    ]
    return [*heading_lines, *explicit_fallback]


def _event_card_lookup_key(title: str) -> str:
    """Build a stable lookup key for event-card title matching."""
    normalized = _strip_accents_compat(title or "").lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _event_has_note_like_description(value: str) -> bool:
    """Return whether an event description is actually a generic note/warning."""
    normalized = _strip_accents_compat(_strip_markdown_formatting(value or "")).lower()
    note_markers = (
        "nota:",
        "notas uteis",
        "helpful notes",
        "convem verificar",
        "convém verificar",
        "pagina oficial",
        "página oficial",
        "alteracoes de horarios",
        "alterações de horários",
        "recorrentes",
        "remain active this week",
        "changes to times/prices",
    )
    return any(marker in normalized for marker in note_markers)


def _clean_event_field_value(value: str, field_key: str) -> str:
    """Strip duplicated label prefixes and stray markdown from parsed event values."""
    cleaned = (value or "").strip()
    if not cleaned:
        return ""

    label_aliases = {
        "description": ("descricao", "descrição", "description", "brief description"),
        "address": ("morada", "address", "localizacao", "localização", "location", "venue"),
        "when": ("quando", "when", "data/hora", "date/time"),
        "duration": ("duracao", "duração", "duration"),
        "category": ("categoria", "category"),
        "price": ("preco", "preço", "price"),
        "schedule": ("horarios", "horários", "schedule"),
        "highlights": ("destaques", "highlights"),
    }
    aliases = label_aliases.get(field_key, ())
    if aliases:
        cleaned = re.sub(
            rf"^(?:\*\*)?(?:{'|'.join(re.escape(alias) for alias in aliases)})(?:\*\*)?:?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    cleaned = cleaned.strip()
    cleaned = re.sub(r"^\*+\s*", "", cleaned)
    if field_key in {"when", "duration", "category", "price", "schedule", "highlights", "description"}:
        cleaned = _strip_markdown_formatting(cleaned).strip()
    return cleaned


def _event_card_icon(title: str, category: str = "", current_icon: str = "") -> str:
    """Choose a more representative emoji for event-card headings."""
    haystack = _strip_accents_compat(f"{title} {category}").lower()
    icon_rules = [
        (("film", "cinema", "movie", "festival de cinema"), "🎬"),
        (("music", "concert", "fado", "jazz", "dj", "live music"), "🎵"),
        (("market", "mercado", "feira", "handicraft", "craft"), "🛍️"),
        (("guard", "guarda", "military", "gnr"), "🪖"),
        (("triathlon", "marathon", "grand prix", "athletics", "sport", "desporto"), "🏅"),
        (("monument", "site", "heritage", "museum", "museu", "palace", "palacio", "palácio"), "🏛️"),
        (("theatre", "teatro", "opera", "dance", "danca", "dança"), "🎭"),
        (("gastronomy", "food", "wine", "culinary", "gastronomia"), "🍽️"),
    ]
    for keywords, icon in icon_rules:
        if any(keyword in haystack for keyword in keywords):
            return icon
    if current_icon and current_icon.strip() and current_icon not in {"📅", "🎭"}:
        return current_icon.strip()
    return "🎭"


def _strip_event_card_separators(text: str) -> str:
    """Remove horizontal-rule separators from event-card output to avoid setext-heading glitches."""
    lines = [line for line in (text or "").splitlines() if line.strip() != "---"]
    return clean_newlines("\n".join(lines)).strip()


def _build_researcher_event_intro_lines(
    events: list[dict[str, object]],
    user_query: str,
    language: str = "en",
) -> list[str]:
    """Create a deterministic intro for researcher event responses when the LLM omits it."""
    if not events:
        return []

    normalized_query = _strip_accents_compat(user_query or "").lower()
    is_pt = language == "pt"
    general_markers = (
        "eventos", "events", "esta semana", "this week", "fim de semana", "weekend",
        "concertos", "concerts", "museus", "cultura", "culture", "grandes eventos",
    )
    music_markers = ("musica", "música", "music", "ao vivo", "live")
    one_event = len(events) == 1

    if one_event and not any(marker in normalized_query for marker in general_markers):
        title = str(events[0].get("title") or "").strip()
        when_value = str(events[0].get("when") or "").strip()
        if is_pt:
            if when_value:
                return [
                    "### 🎭 Evento Cultural",
                    f"O evento **{title}** está agendado para **{when_value}**. Todas as informações disponíveis que tenho são:",
                ]
            return [
                "### 🎭 Evento Cultural",
                f"Aqui estão as informações disponíveis sobre **{title}**:",
            ]
        if when_value:
            return [
                "### 🎭 Cultural Event",
                f"The event **{title}** is scheduled for **{when_value}**. Here is all the information I have available:",
            ]
        return [
            "### 🎭 Cultural Event",
            f"Here is the information I have available about **{title}**:",
        ]

    if is_pt:
        if any(marker in normalized_query for marker in music_markers) and any(marker in normalized_query for marker in ("fim de semana", "weekend")):
            return [
                "### 🎭 Eventos Culturais",
                "Aqui tens uma seleção de eventos de música ao vivo para este fim de semana em Lisboa:",
            ]
        if "esta semana" in normalized_query or "this week" in normalized_query:
            return [
                "### 🎭 Eventos Culturais",
                "Aqui tens uma seleção de eventos culturais e de grande visibilidade esta semana em Lisboa:",
            ]
        return [
            "### 🎭 Eventos Culturais",
            "Aqui tens os principais eventos culturais que encontrei em Lisboa:",
        ]

    if any(marker in normalized_query for marker in music_markers) and any(marker in normalized_query for marker in ("weekend", "fim de semana")):
        return [
            "### 🎭 Cultural Events",
            "Here is a selection of live-music events for this weekend in Lisbon:",
        ]
    if "this week" in normalized_query or "esta semana" in normalized_query:
        return [
            "### 🎭 Cultural Events",
            "Here is a selection of high-visibility cultural events in Lisbon this week:",
        ]
    return [
        "### 🎭 Cultural Events",
        "Here are the main cultural events I found in Lisbon:",
    ]


def _is_researcher_place_meta_line(text: str) -> bool:
    """Return whether a line is a raw place-summary line that should not surface above canonical cards."""
    normalized = _strip_accents_compat(_strip_markdown_formatting(text or "")).lower().strip()
    return bool(
        re.match(r"^(?:found|encontrei)\s+\d+\s+(?:places|place|locais|atracoes|atrações)", normalized)
        or normalized.startswith(("places/attractions in lisbon", "locais em lisboa", "atracoes em lisboa", "atrações em lisboa"))
    )


def _build_researcher_place_intro_lines(
    cards: list[dict[str, object]],
    user_query: str,
    language: str = "en",
) -> list[str]:
    """Create a deterministic intro for researcher place responses when the LLM omits it."""
    if not cards:
        return []

    normalized_query = _strip_accents_compat(user_query or "").lower()
    is_pt = language == "pt"
    title = str(cards[0].get("title") or "").strip()
    category = _strip_accents_compat(str(cards[0].get("category") or "")).lower()
    general_markers = (
        "museus", "museums", "restaurants", "restaurantes", "atrações", "atracoes",
        "places", "locais", "best", "top", "perto", "near", "onde", "where",
    )
    museum_markers = ("museum", "museu", "monument", "monumento", "palacio", "palácio")
    dining_markers = ("restaurant", "restaurante", "seafood", "marisco", "food", "gastronomia", "dining")

    if len(cards) == 1 and not any(marker in normalized_query for marker in general_markers):
        if is_pt:
            return [
                "### 📍 Local em Lisboa",
                f"Aqui estão as informações disponíveis sobre **{title}**:",
            ]
        return [
            "### 📍 Place in Lisbon",
            f"Here is the information I have available about **{title}**:",
        ]

    if is_pt:
        if any(marker in normalized_query for marker in dining_markers) or "restaurant" in category or "restaurante" in category:
            return ["### 🍽️ Locais Recomendados", "Aqui tens locais de restauração em Lisboa que correspondem ao que pediste:"]
        if any(marker in normalized_query for marker in museum_markers) or any(marker in category for marker in museum_markers):
            return ["### 🏛️ Locais Recomendados", "Aqui tens uma seleção de museus e locais culturais em Lisboa que correspondem ao pedido:"]
        return ["### 📍 Locais Recomendados", "Aqui tens os principais locais que encontrei em Lisboa para o que pediste:"]

    if any(marker in normalized_query for marker in dining_markers) or "restaurant" in category:
        return ["### 🍽️ Recommended Places", "Here are dining spots in Lisbon that match your request:"]
    if any(marker in normalized_query for marker in museum_markers) or any(marker in category for marker in museum_markers):
        return ["### 🏛️ Recommended Places", "Here is a selection of museums and cultural places in Lisbon that match your request:"]
    return ["### 📍 Recommended Places", "Here are the main places I found in Lisbon for your request:"]


def _parse_structured_event_cards(text: str, language: str = "en") -> tuple[list[str], list[dict[str, object]], str]:
    """Parse structured event-card markdown into intro lines, event dicts, and a source line."""
    if not text:
        return [], [], ""

    is_pt = language == "pt"
    localized_label_map = {
        "quando": "when",
        "data/hora": "when",
        "date/time": "when",
        "when": "when",
        "duração": "duration",
        "duracao": "duration",
        "duration": "duration",
        "categoria": "category",
        "category": "category",
        "descrição": "description",
        "descricao": "description",
        "description": "description",
        "morada": "address",
        "address": "address",
        "localização": "address",
        "localizacao": "address",
        "location": "address",
        "venue": "address",
        "preço": "price",
        "preco": "price",
        "price": "price",
        "horários": "schedule",
        "horarios": "schedule",
        "schedule": "schedule",
        "destaques": "highlights",
        "highlights": "highlights",
        "mais detalhes": "details_url",
        "more details": "details_url",
        "comprar bilhetes": "tickets_url",
        "buy tickets": "tickets_url",
        "bilhetes": "tickets_url",
        "tickets": "tickets_url",
    }
    heading_re = re.compile(r"^###\s+(?P<emoji>[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+)?\s*(?P<title>.+?)\s*$")
    bullet_re = re.compile(
        r"^\s*[-*•]?\s*(?P<emoji>[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+)?\s*(?:\*\*(?P<label>[^*]+?)\*\*:?)?\s*(?P<value>.+?)\s*$"
    )
    inline_section_heading_re = re.compile(
        r"^(?P<emoji>[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+)\s+\*\*(?P<title>.+?)\*\*\s*$"
    )

    def _new_event(icon: str, title: str) -> dict[str, object]:
        return {
            "icon": icon,
            "title": title,
            "when": "",
            "duration": "",
            "category": "",
            "description": "",
            "address": "",
            "price": "",
            "schedule": "",
            "highlights": "",
            "details_url": "",
            "tickets_url": "",
            "extra_lines": [],
        }

    def _assign_line(line: str, event: dict[str, object]) -> None:
        for segment in re.split(
            r"\s+(?:\|\||--|—|–|\||- )\s+(?=(?:[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]|https?://|\*\*))",
            re.sub(r"^(?:[-*•]\s+)?", "", line.strip()),
        ):
            stripped = segment.strip()
            if not stripped or stripped == "---":
                continue
            if stripped.startswith(("⚠️", "🔎")) or _is_researcher_event_meta_line(stripped):
                continue
            if stripped.startswith("🌐"):
                event["details_url"] = _extract_first_url(stripped) or stripped.removeprefix("🌐").strip()
                continue
            if stripped.startswith("🎟️"):
                event["tickets_url"] = _extract_first_url(stripped) or stripped.removeprefix("🎟️").strip()
                continue
            if stripped.startswith("🔗"):
                event["details_url"] = _extract_first_url(stripped) or stripped.removeprefix("🔗").strip()
                continue
            match = bullet_re.match(stripped)
            if not match:
                if stripped not in event["extra_lines"]:
                    event["extra_lines"].append(stripped)
                continue
            emoji = (match.group("emoji") or "").strip()
            label = _strip_accents_compat((match.group("label") or "").strip().rstrip(":")).lower()
            value = (match.group("value") or "").strip()
            if label:
                normalized_key = localized_label_map.get(label)
            else:
                normalized_key = {
                    "📍": "address",
                    "🗓️": "when",
                    "📅": "when",
                    "⏱️": "duration",
                    "📂": "category",
                    "📝": "description",
                    "💰": "price",
                    "💶": "price",
                    "🕐": "schedule",
                    "✨": "highlights",
                }.get(emoji)

            if normalized_key == "details_url":
                event["details_url"] = _extract_first_url(value) or value
                continue
            if normalized_key == "tickets_url":
                event["tickets_url"] = _extract_first_url(value) or value
                continue
            if normalized_key:
                cleaned_value = _clean_event_field_value(value, normalized_key)
                if normalized_key == "description" and _event_has_note_like_description(cleaned_value):
                    return
                event[normalized_key] = cleaned_value
                continue
            if stripped not in event["extra_lines"]:
                event["extra_lines"].append(stripped)

    intro_lines: list[str] = []
    events: list[dict[str, object]] = []
    source_line = ""
    current_event: Optional[dict[str, object]] = None

    def _flush() -> None:
        nonlocal current_event
        if not current_event:
            return
        if current_event.get("description") and _event_has_note_like_description(str(current_event["description"])):
            current_event["description"] = ""
        if any(str(current_event.get(key) or "").strip() for key in ("description", "address", "when", "category", "price", "details_url", "tickets_url")):
            events.append(current_event)
        current_event = None

    for raw_line in (text or "").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped == "---":
            continue
        if _is_researcher_event_meta_line(stripped):
            continue
        if _SOURCE_LINE_RE.match(stripped):
            _flush()
            source_line = stripped
            continue
        inline_heading_match = inline_section_heading_re.match(stripped)
        if inline_heading_match:
            normalized_inline_title = _event_card_lookup_key(inline_heading_match.group("title"))
            if normalized_inline_title in {"eventos culturais", "cultural events", "notas uteis", "helpful notes"}:
                _flush()
                if normalized_inline_title in {"eventos culturais", "cultural events"}:
                    intro_lines.append(f"### {inline_heading_match.group('emoji').strip()} {inline_heading_match.group('title').strip()}")
                continue
        heading_match = heading_re.match(stripped)
        if heading_match:
            title = heading_match.group("title").strip()
            normalized_title = _event_card_lookup_key(title)
            if normalized_title in {"eventos culturais", "cultural events", "notas uteis", "helpful notes"}:
                _flush()
                intro_lines.append(stripped)
                continue
            _flush()
            current_event = _new_event((heading_match.group("emoji") or "🎭").strip() or "🎭", title)
            continue
        if current_event is None:
            if not (stripped.startswith(("⚠️", "🔎", "💡")) or _event_has_note_like_description(stripped)):
                intro_lines.append(stripped)
            continue
        _assign_line(raw_line, current_event)

    _flush()
    if source_line:
        source_line = canonicalize_visitlisboa_source_line(source_line, language="pt" if is_pt else "en")
    return intro_lines, events, source_line


def reconcile_researcher_event_response(
    text: str,
    worker_text: str,
    language: str = "en",
    user_query: str = "",
) -> str:
    """Rehydrate event-card metadata lost in QA/final formatting using the grounded worker output."""
    if infer_researcher_source_kind(user_query=user_query, text=text) != "events":
        return text
    worker_canonical = format_researcher_event_cards(worker_text, language=language, user_query=user_query)
    primary_intro, primary_events, primary_source = _parse_structured_event_cards(text, language=language)
    fallback_intro, fallback_events, fallback_source = _parse_structured_event_cards(worker_canonical, language=language)
    if not primary_events:
        return _strip_event_card_separators(worker_canonical or text)
    fallback_by_title = {
        _event_card_lookup_key(str(event.get("title") or "")): event
        for event in fallback_events
    }
    merged_events: list[dict[str, object]] = []
    for event in primary_events:
        merged = dict(event)
        fallback = fallback_by_title.get(_event_card_lookup_key(str(event.get("title") or "")))
        if fallback:
            for key in ("when", "duration", "category", "address", "price", "schedule", "highlights", "details_url", "tickets_url"):
                if not str(merged.get(key) or "").strip() and str(fallback.get(key) or "").strip():
                    merged[key] = fallback.get(key)
            if (
                not str(merged.get("description") or "").strip()
                or _event_has_note_like_description(str(merged.get("description") or ""))
            ) and str(fallback.get("description") or "").strip():
                merged["description"] = fallback.get("description")
            if not str(merged.get("icon") or "").strip() or str(merged.get("icon")) in {"📅", "🎭"}:
                merged["icon"] = fallback.get("icon") or merged.get("icon")
            if not merged.get("extra_lines") and fallback.get("extra_lines"):
                merged["extra_lines"] = list(fallback.get("extra_lines") or [])
        merged_events.append(merged)

    intro_lines = _select_researcher_specific_lookup_intro(primary_intro, fallback_intro)
    if not intro_lines:
        intro_lines = [line for line in primary_intro if not _event_has_note_like_description(line)] or [line for line in fallback_intro if not _event_has_note_like_description(line)]
    if not intro_lines:
        intro_lines = _build_researcher_event_intro_lines(merged_events, user_query=user_query, language=language)
    source_line = primary_source or fallback_source
    rendered_lines: list[str] = []
    for line in intro_lines:
        rendered_lines.append(line)
    if rendered_lines:
        rendered_lines.append("")

    description_label = "Descrição" if language == "pt" else "Description"
    date_label = "Data/Hora" if language == "pt" else "Date/Time"
    duration_label = "Duração" if language == "pt" else "Duration"
    category_label = "Categoria" if language == "pt" else "Category"
    address_label = "Morada" if language == "pt" else "Address"
    price_label = "Preço" if language == "pt" else "Price"
    schedule_label = "Horários" if language == "pt" else "Schedule"
    highlights_label = "Destaques" if language == "pt" else "Highlights"
    details_label = "Mais detalhes" if language == "pt" else "More details"
    tickets_label = "Bilhetes" if language == "pt" else "Tickets"

    for event in merged_events:
        icon = _event_card_icon(str(event.get("title") or ""), str(event.get("category") or ""), str(event.get("icon") or ""))
        rendered_lines.append(f"### {icon} {event['title']}")
        rendered_lines.append("")
        if event.get("description"):
            rendered_lines.append(f"- 📝 **{description_label}:** {event['description']}")
        if event.get("address"):
            address_value = str(event["address"]).strip()
            if "](" not in address_value:
                address_value = f"[{address_value}]({_gmaps_link(address_value)})"
            rendered_lines.append(f"- 📍 **{address_label}:** {address_value}")
        if event.get("when"):
            rendered_lines.append(f"- 📅 **{date_label}:** {event['when']}")
        if event.get("duration"):
            rendered_lines.append(f"- ⏱️ **{duration_label}:** {event['duration']}")
        if event.get("category"):
            rendered_lines.append(f"- 📂 **{category_label}:** {event['category']}")
        if event.get("price"):
            rendered_lines.append(f"- 💰 **{price_label}:** {event['price']}")
        if event.get("schedule"):
            rendered_lines.append(f"- 🕐 **{schedule_label}:** {event['schedule']}")
        if event.get("highlights"):
            rendered_lines.append(f"- ✨ **{highlights_label}:** {event['highlights']}")
        if event.get("details_url"):
            rendered_lines.append(f"- 🌐 [{details_label}]({str(event['details_url']).strip()})")
        if event.get("tickets_url"):
            rendered_lines.append(f"- 🎟️ [{tickets_label}]({str(event['tickets_url']).strip()})")
        for extra_line in list(event.get("extra_lines") or []):
            if extra_line and not _event_has_note_like_description(str(extra_line)) and not str(extra_line).strip().startswith(("⚠️", "🔎", "💡")):
                rendered_lines.append(f"- {str(extra_line).strip()}")
        rendered_lines.append("")

    if source_line:
        rendered_lines.append(source_line)
    return _strip_event_card_separators(clean_newlines("\n".join(rendered_lines)).strip())


def reconcile_researcher_place_response(
    text: str,
    worker_text: str,
    language: str = "en",
    user_query: str = "",
) -> str:
    """Rehydrate canonical place cards when QA or synthesis collapses grounded fields."""
    if infer_researcher_source_kind(user_query=user_query, text=text) != "places":
        return text

    worker_canonical = format_researcher_card(worker_text, language=language, user_query=user_query)
    if not worker_canonical:
        return text

    primary_count = _count_structured_place_cards(text)
    fallback_count = _count_structured_place_cards(worker_canonical)
    if primary_count == 0 and fallback_count > 0:
        return worker_canonical
    if primary_count <= 0:
        return text
    if _place_response_missing_required_fields(text, language, primary_count):
        return worker_canonical
    return text


def format_researcher_event_cards(text: str, language: str = "en", user_query: str = "") -> str:
    """Normalize ranked researcher event results into canonical markdown cards."""
    if not text or infer_researcher_source_kind(user_query=user_query, text=text) != "events":
        return text

    is_pt = language == "pt"
    date_label = "Data/Hora" if is_pt else "Date/Time"
    duration_label = "Duração" if is_pt else "Duration"
    category_label = "Categoria" if is_pt else "Category"
    description_label = "Descrição" if is_pt else "Description"
    address_label = "Morada" if is_pt else "Address"
    price_label = "Preço" if is_pt else "Price"
    schedule_label = "Horários" if is_pt else "Schedule"
    highlights_label = "Destaques" if is_pt else "Highlights"
    details_label = "Mais detalhes" if is_pt else "More details"
    tickets_label = "Bilhetes" if is_pt else "Tickets"
    default_icon = "📅"

    localized_label_map = {
        "quando": "when",
        "data/hora": "when",
        "date/time": "when",
        "date": "when",
        "when": "when",
        "duração": "duration",
        "duracao": "duration",
        "duration": "duration",
        "categoria": "category",
        "category": "category",
        "descrição": "description",
        "descricao": "description",
        "breve descrição": "description",
        "breve descricao": "description",
        "description": "description",
        "brief description": "description",
        "morada": "address",
        "address": "address",
        "localização": "address",
        "localizacao": "address",
        "location": "address",
        "local": "address",
        "venue": "address",
        "preço": "price",
        "preco": "price",
        "price": "price",
        "horários": "schedule",
        "horarios": "schedule",
        "schedule": "schedule",
        "destaques": "highlights",
        "highlights": "highlights",
        "mais detalhes": "details_url",
        "more details": "details_url",
        "comprar bilhetes": "tickets_url",
        "buy tickets": "tickets_url",
        "bilhetes": "tickets_url",
        "tickets": "tickets_url",
    }

    start_re = re.compile(
        r"^\s*(?:(?:\*\*)?(?P<num>\d+)\.?(?:\*\*)|[-*•]|###)\s+(?P<rest>.+)$"
    )
    field_re = re.compile(
        r"^(?P<emoji>[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+)?\s*\*\*(?P<label>[^*]+?)\*\*:?[ \t]*(?P<value>.+)$"
    )

    def _parse_start_line(line: str) -> Optional[tuple[str, str]]:
        stripped_line = line.strip()
        match = start_re.match(stripped_line)
        if match:
            rest = match.group("rest").strip()
        else:
            rest = re.sub(r"^(?:\*\*)?\d+\.?(?:\*\*)?\s+", "", stripped_line).strip()
            if rest == stripped_line:
                rest = re.sub(r"^(?:[-*•]|###)\s+", "", stripped_line).strip()
            if not rest or rest == stripped_line:
                return None
        is_markdown_heading = stripped_line.startswith("### ")
        if is_markdown_heading:
            title_match = re.match(
                r"^(?P<emoji>[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+)?\s*(?:\*\*(?P<title_bold>.+?)\*\*|(?P<title_plain>.+?))\s*$",
                rest,
            )
        else:
            title_match = re.match(
                r"^(?P<emoji>[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+)?\s*\*\*(?P<title_bold>.+?)\*\*\s*$",
                rest,
            )
        if not title_match:
            return None
        emoji = (title_match.group("emoji") or default_icon).strip() or default_icon
        title = (title_match.group("title_bold") or title_match.group("title_plain") or "").strip()
        if not title:
            return None
        return emoji, title

    def _normalize_segments(raw_line: str) -> list[str]:
        stripped = raw_line.strip()
        if not stripped:
            return []
        base = re.sub(r"^(?:[-*•]\s+)?", "", stripped)
        return re.split(
            r"\s+(?:\|\||--|—|–|\||-)\s+(?=(?:[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]|https?://|\*\*))",
            base,
        )

    def _new_event(icon: str, title: str) -> dict[str, object]:
        return {
            "icon": icon,
            "title": title,
            "when": "",
            "duration": "",
            "category": "",
            "description": "",
            "address": "",
            "price": "",
            "schedule": "",
            "highlights": "",
            "details_url": "",
            "tickets_url": "",
            "extra_lines": [],
        }

    def _extract_url(value: str) -> str:
        url_match = re.search(r"https?://\S+", value)
        return url_match.group(0).rstrip(").,;") if url_match else ""

    def _assign_segment(segment: str, event: dict[str, object]) -> None:
        stripped = segment.strip()
        if not stripped:
            return

        stripped = re.sub(
            r"^(?P<emoji>[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+)\s*(?P<label>[^*\[][^:]+?)\*\*:\s*\*\*(?P<value>.+)$",
            lambda match: f"{match.group('emoji')} **{match.group('label').strip()}:** {match.group('value').strip()}",
            stripped,
        )
        stripped = re.sub(
            r"^(?P<emoji>[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+)\s*(?P<label>[^*\[]+?):\*\*\s*\*\*(?P<value>.+)$",
            lambda match: f"{match.group('emoji')} **{match.group('label').strip()}:** {match.group('value').strip()}",
            stripped,
        )

        field_match = field_re.match(stripped)
        if field_match:
            label_key = _strip_accents_compat(field_match.group("label").strip().rstrip(":")).lower()
            value = field_match.group("value").strip()
            normalized_key = localized_label_map.get(label_key)
            if normalized_key == "details_url":
                event["details_url"] = _extract_url(value) or value
                return
            if normalized_key == "tickets_url":
                event["tickets_url"] = _extract_url(value) or value
                return
            if normalized_key:
                event[normalized_key] = _clean_event_field_value(value, normalized_key)
                return

        plain = re.sub(r"^(?:[-*•]\s+)?", "", stripped)
        if plain.startswith("🔗"):
            event["details_url"] = _extract_url(plain) or plain.removeprefix("🔗").strip()
            return
        if plain.startswith("🎟️"):
            event["tickets_url"] = _extract_url(plain) or plain.removeprefix("🎟️").strip()
            return
        if plain.startswith("📍"):
            event["address"] = _clean_event_field_value(plain.removeprefix("📍").strip(), "address")
            return
        if plain.startswith("🗓️") or plain.startswith("📅"):
            when_value = plain.lstrip("🗓️📅").strip()
            when_value = re.sub(
                r"^(?:\*\*)?(?:Quando|When|Data/Hora|Date/Time)(?:\*\*)?:?\s*",
                "",
                when_value,
                flags=re.IGNORECASE,
            )
            event["when"] = _clean_event_field_value(_strip_markdown_formatting(when_value).strip(), "when")
            return
        if plain.startswith("⏱️"):
            event["duration"] = _clean_event_field_value(plain.removeprefix("⏱️").strip(), "duration")
            return
        if plain.startswith("📂"):
            event["category"] = _clean_event_field_value(plain.removeprefix("📂").strip(), "category")
            return
        if plain.startswith("📝"):
            event["description"] = _clean_event_field_value(plain.removeprefix("📝").strip(), "description")
            return
        if plain.startswith("💰"):
            event["price"] = _clean_event_field_value(plain.removeprefix("💰").strip(), "price")
            return
        if plain.startswith("🕐"):
            event["schedule"] = _clean_event_field_value(plain.removeprefix("🕐").strip(), "schedule")
            return
        if plain.startswith("✨"):
            event["highlights"] = _clean_event_field_value(plain.removeprefix("✨").strip(), "highlights")
            return
        bare_url = _extract_url(plain)
        if bare_url:
            if not event["details_url"]:
                event["details_url"] = bare_url
            else:
                event["extra_lines"].append(plain)
            return
        if not str(event.get("description") or "").strip():
            cleaned_plain = _clean_event_field_value(plain, "description")
            if not _event_has_note_like_description(cleaned_plain):
                event["description"] = cleaned_plain
            return
        event["extra_lines"].append(plain)

    def _render_link(label: str, url: str) -> str:
        cleaned_url = (url or "").strip()
        if not cleaned_url:
            return ""
        if cleaned_url.startswith("[") and "](" in cleaned_url:
            return cleaned_url
        return f"[{label}]({cleaned_url})"

    def _flush_event(event: Optional[dict[str, object]], output_lines: list[str]) -> None:
        if not event:
            return
        if output_lines and output_lines[-1] != "":
            output_lines.append("")
        icon = _event_card_icon(str(event.get("title") or ""), str(event.get("category") or ""), str(event.get("icon") or ""))
        output_lines.append(f"### {icon} {event['title']}")
        output_lines.append("")

        if event["description"] and not _event_has_note_like_description(str(event["description"])):
            output_lines.append(f"- 📝 **{description_label}:** {event['description']}")
        if event["address"]:
            address_value = str(event["address"]).strip()
            if "](" not in address_value:
                address_value = f"[{address_value}]({_gmaps_link(address_value)})"
            output_lines.append(f"- 📍 **{address_label}:** {address_value}")
        if event["when"]:
            output_lines.append(f"- 📅 **{date_label}:** {event['when']}")
        if event["duration"]:
            output_lines.append(f"- ⏱️ **{duration_label}:** {event['duration']}")
        if event["category"]:
            output_lines.append(f"- 📂 **{category_label}:** {event['category']}")
        if event["price"]:
            output_lines.append(f"- 💰 **{price_label}:** {event['price']}")
        if event["schedule"]:
            output_lines.append(f"- 🕐 **{schedule_label}:** {event['schedule']}")
        if event["highlights"]:
            output_lines.append(f"- ✨ **{highlights_label}:** {event['highlights']}")
        if event["details_url"]:
            output_lines.append(f"- 🌐 {_render_link(details_label, str(event['details_url']))}")
        if event["tickets_url"]:
            output_lines.append(f"- 🎟️ {_render_link(tickets_label, str(event['tickets_url']))}")
        for extra_line in event["extra_lines"]:
            if not _event_has_note_like_description(str(extra_line)) and not str(extra_line).strip().startswith(("⚠️", "🔎", "💡")):
                output_lines.append(f"- {str(extra_line)}")
        output_lines.append("")

    lines = text.splitlines()
    output_lines: list[str] = []
    current_event: Optional[dict[str, object]] = None
    transformed = False
    skipping_summary_block = False

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue

        if stripped == "---":
            continue

        if current_event is not None and stripped.startswith("### "):
            heading_body = re.sub(r"^###\s+", "", stripped).strip()
            normalized_heading_body = _strip_accents_compat(
                _strip_markdown_formatting(_strip_leading_section_emoji(heading_body))
            ).lower()
            if "notas uteis" in normalized_heading_body or "helpful notes" in normalized_heading_body:
                _flush_event(current_event, output_lines)
                current_event = None
                skipping_summary_block = True
                transformed = True
                continue
            field_heading_prefixes = (
                "data/hora",
                "date/time",
                "preco",
                "preço",
                "price",
                "duration",
                "duracao",
                "duração",
                "more details",
                "mais detalhes",
                "tickets",
                "bilhetes",
            )
            if heading_body.startswith(("🌐 ", "🎟️ ")):
                _assign_segment(heading_body, current_event)
                transformed = True
                continue
            if any(normalized_heading_body.startswith(prefix) for prefix in field_heading_prefixes):
                _assign_segment(heading_body, current_event)
                transformed = True
                continue

        if _SOURCE_LINE_RE.match(stripped):
            _flush_event(current_event, output_lines)
            current_event = None
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            output_lines.append(stripped)
            continue

        start = _parse_start_line(stripped)
        if start:
            _flush_event(current_event, output_lines)
            icon, title = start
            normalized_title = _strip_accents_compat(title).lower()
            if normalized_title in {"resumo da pesquisa", "search summary"}:
                current_event = None
                skipping_summary_block = True
                transformed = True
                continue
            if normalized_title in {"eventos culturais", "cultural events"}:
                current_event = None
                skipping_summary_block = False
                output_lines.append(f"### {icon} {title}" if stripped.startswith("### ") else stripped)
                transformed = True
                continue
            skipping_summary_block = False
            current_event = _new_event(icon, title)
            transformed = True
            continue

        if skipping_summary_block:
            continue

        if current_event is None and _is_researcher_event_meta_line(stripped):
            continue

        section_heading_match = re.match(
            r"^(?P<emoji>[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+)\s+\*\*(?P<title>.+?)\*\*\s*$",
            stripped,
        )
        if current_event is None and section_heading_match:
            normalized_section_title = _strip_accents_compat(section_heading_match.group("title")).lower().strip()
            if normalized_section_title in {"eventos culturais", "cultural events"}:
                output_lines.append(f"### {section_heading_match.group('emoji').strip()} {section_heading_match.group('title').strip()}")
                transformed = True
                continue

        if current_event is None:
            output_lines.append(stripped)
            continue

        for segment in _normalize_segments(raw_line):
            _assign_segment(segment, current_event)

    _flush_event(current_event, output_lines)

    if not transformed:
        return text
    return _strip_event_card_separators(clean_newlines("\n".join(output_lines)).strip())


def format_researcher_card(text: str, language: str = "en", user_query: str = "") -> str:
    """Normalize ranked researcher place results into canonical markdown cards."""
    if not text or "**" not in text:
        return text
    if "Lisboa Aberta" in text or "dados.cm-lisboa.pt" in text:
        return text
    if infer_researcher_source_kind(user_query=user_query, text=text) != "places":
        return text

    labels = _researcher_card_labels(language)
    lines = text.splitlines()
    output_lines: list[str] = []
    rendered_cards: list[dict[str, object]] = []
    saw_intro_text = False
    transformed = False
    current_card: Optional[dict[str, object]] = None

    def flush_card() -> None:
        nonlocal current_card
        if not current_card:
            return

        card_lines = [f"### {current_card['emoji']} {current_card['title']}", ""]
        field_order = [
            ("description", "📝"),
            ("category", "📂"),
            ("address", "📍"),
            ("phone", "📞"),
            ("rating", "⭐"),
            ("price", "💶"),
            ("today", "🕐"),
            ("hours", "🕐"),
            ("website", "🌐"),
            ("tickets", "🎟️"),
            ("distance", "📏"),
            ("coordinates", "🗺️"),
        ]

        for key, emoji in field_order:
            value = str(current_card.get(key) or "").strip()
            if not value:
                continue
            label = labels[key]
            if key == "address" and "](" not in value:
                value = f"[{value}]({_gmaps_link(value)})"
            elif key == "phone":
                value = linkify_phone_numbers(value)
            elif key in {"website", "tickets"}:
                value = _render_researcher_link_value(value, label)
            card_lines.append(f"- {emoji} **{label}:** {value}")

        extra_lines = current_card.get("extra_lines", [])
        if isinstance(extra_lines, list):
            for extra_line in extra_lines:
                normalized_extra = str(extra_line).strip()
                if not normalized_extra:
                    continue
                card_lines.append(normalized_extra if normalized_extra.startswith(("- ", "* ")) else f"- {normalized_extra}")

        output_lines.extend(card_lines)
        output_lines.append("")
        rendered_cards.append(dict(current_card))
        current_card = None

    for raw_line in lines:
        stripped = raw_line.strip()
        start_match = _RESEARCHER_CARD_START_RE.match(stripped)

        if start_match:
            flush_card()
            raw_title = start_match.group("title").strip()
            title = raw_title.split(" | ", 1)[0].strip()
            current_card = {
                "emoji": start_match.group("emoji"),
                "title": title,
                "description": "",
                "category": "",
                "address": "",
                "phone": "",
                "rating": "",
                "price": "",
                "website": "",
                "tickets": "",
                "today": "",
                "hours": "",
                "distance": "",
                "coordinates": "",
                "extra_lines": [],
            }
            transformed = True
            continue

        if not current_card:
            if not _is_researcher_place_meta_line(stripped):
                output_lines.append(raw_line)
                if stripped and not _SOURCE_LINE_RE.match(stripped):
                    saw_intro_text = True
            continue

        if not stripped:
            continue
        if _SOURCE_LINE_RE.match(stripped):
            flush_card()
            output_lines.append(raw_line)
            continue

        content_line = re.sub(r"^(?:[-*]\s+)?", "", stripped)
        normalized_line = re.sub(r"^[📂📍🕐⭐📞🔗🌐💶🎟️📝🗺️📏]\s*", "", content_line).strip()
        field_match = re.match(r"^\*\*(?P<label>[^*]+?)\*\*:?[ \t]*(?P<value>.*)$", normalized_line)

        label = ""
        value = normalized_line
        if field_match:
            label = field_match.group("label").strip().rstrip(":")
            value = field_match.group("value").strip()

        label_key = _strip_accents_compat(label).lower()
        value_lower = _strip_accents_compat(value).lower()

        if label_key in {"category", "categoria"}:
            current_card["category"] = value
        elif label_key in {"description", "descricao", "descrição"}:
            current_card["description"] = value
        elif label_key in {"address", "morada", "location", "localizacao", "localização"}:
            current_card["address"] = value
        elif label_key in {"phone", "telefone", "contacto", "contact"}:
            current_card["phone"] = value
        elif label_key in {"tripadvisor", "rating", "avaliacao", "avaliação", "reviews", "avaliacoes", "avaliações"}:
            current_card["rating"] = value
        elif label_key in {"price", "preco", "preço", "prices", "precos", "preços"}:
            current_card["price"] = value
        elif label_key in {"website", "site oficial", "official page", "url"}:
            current_card["website"] = value or normalized_line
        elif label_key in {"tickets", "bilhetes", "buy tickets", "comprar bilhetes"}:
            current_card["tickets"] = value or normalized_line
        elif label_key in {"today", "hoje"}:
            current_card["today"] = value
        elif label_key in {"hours", "horario", "horário", "opening hours"}:
            current_card["hours"] = value
        elif label_key in {"distance", "distancia", "distância"}:
            current_card["distance"] = value
        elif label_key in {"coordinates", "coordenadas"}:
            current_card["coordinates"] = value
        elif normalized_line.startswith("http") or "visitlisboa.com" in value_lower:
            current_card["website"] = normalized_line
        elif content_line.startswith("📞") or re.search(r"(?:\+?351|00351)\s*\d{3}\s*\d{3}\s*\d{3}", normalized_line):
            current_card["phone"] = normalized_line
        elif content_line.startswith("📍"):
            current_card["address"] = value if field_match else normalized_line
        elif content_line.startswith("⭐"):
            current_card["rating"] = value if field_match else normalized_line
        elif content_line.startswith("🕐"):
            current_card["today"] = value if field_match else normalized_line
        elif not str(current_card.get("description") or "").strip():
            current_card["description"] = normalized_line
        else:
            current_card["extra_lines"].append(normalized_line)

    flush_card()

    if not transformed:
        return text
    if not saw_intro_text:
        intro_lines = _build_researcher_place_intro_lines(rendered_cards, user_query=user_query, language=language)
        if intro_lines:
            output_lines = [*intro_lines, "", *output_lines]
    return clean_newlines("\n".join(output_lines)).strip()


def repair_planner_markdown_contract(text: str, language: str = "en") -> str:
    """Restore the planner markdown contract after generic formatting passes."""
    if not text:
        return text

    is_pt = language == "pt"
    section_icons = {"⛅", "🚇", "📍", "✨", "⚠️", "📝"}
    repaired_lines: list[str] = []
    title_fixed = False

    def itinerary_title_match(value: str) -> Optional[str]:
        body = re.sub(r"^###\s+", "", value).strip()
        normalized_body = _strip_accents_compat(body).lower()
        if any(token in normalized_body for token in ("itiner", "itinerary", "plano", "roteiro")):
            cleaned = re.sub(r"^[^A-Za-zÀ-ÿ0-9]+", "", body).strip(" :-")
            if cleaned:
                return cleaned
        return None

    def canonical_planner_section(value: str) -> Optional[str]:
        normalized_value = _strip_accents_compat(_strip_leading_section_emoji(value)).lower()
        if ("condic" in normalized_value and "meteorolog" in normalized_value) or (
            "weather" in normalized_value and "condition" in normalized_value
        ):
            return f"**⛅ {'Condições Meteorológicas' if is_pt else 'Weather Conditions'}**"
        if "antes de sair" in normalized_value or ("before" in normalized_value and "go" in normalized_value):
            return f"**⛅ {'Antes de Sair' if is_pt else 'Before You Go'}**"
        if ("condic" in normalized_value and "seguran" in normalized_value) or (
            "conditions" in normalized_value and "safety" in normalized_value
        ):
            return f"**⛅ {'Condições e Segurança' if is_pt else 'Conditions and Safety'}**"
        if (
            "como chegar" in normalized_value
            or "desloca" in normalized_value
            or "how to get there" in normalized_value
            or "get around" in normalized_value
        ):
            return f"**🚇 {'Como Chegar e Deslocação' if is_pt else 'How to Get There and Get Around'}**"
        if (
            "sugest" in normalized_value
            or "recomend" in normalized_value
            or "visita" in normalized_value
            or "visit suggestions" in normalized_value
            or "recommendations" in normalized_value
            or "options" in normalized_value
        ) and "janela" not in normalized_value and "window" not in normalized_value:
            return f"**📍 {'Sugestões para a Visita' if is_pt else 'Visit Suggestions'}**"
        if "notas" in normalized_value and "pratic" in normalized_value and "dicas" not in normalized_value and "important" not in normalized_value:
            return f"**✨ {'Notas Práticas' if is_pt else 'Practical Notes'}**"
        if (
            ("dicas" in normalized_value and "notas" in normalized_value)
            or ("dicas" in normalized_value and "pratic" in normalized_value)
            or ("notas" in normalized_value and "important" in normalized_value)
            or ("notas" in normalized_value and "pratic" in normalized_value)
            or "practical tips" in normalized_value
            or "important notes" in normalized_value
            or "final notes" in normalized_value
        ):
            return f"**✨ {'Dicas Práticas e Notas Importantes' if is_pt else 'Practical Tips and Important Notes'}**"
        if normalized_value in {"dicas", "dicas praticas", "dicas práticas", "tips", "practical tips"}:
            return f"**✨ {'Dicas Práticas' if is_pt else 'Practical Tips'}**"
        return None

    def timed_card_icon(title: str) -> str:
        normalized_title = _strip_accents_compat(title).lower()
        if any(
            keyword in normalized_title
            for keyword in (
                "mosteiro",
                "museu",
                "museum",
                "monument",
                "igreja",
                "church",
                "torre",
                "castle",
                "castelo",
                "palacio",
                "palácio",
                "praca",
                "praça",
                "belem",
                "belém",
                "chegada",
            )
        ):
            return "\U0001F3DB\uFE0F"
        if any(keyword in normalized_title for keyword in ("pastel", "nata", "bakery", "pastry")):
            return "\U0001F950"
        if any(keyword in normalized_title for keyword in ("cafe", "café", "coffee", "bar")):
            return "\u2615"
        if any(keyword in normalized_title for keyword in ("almoco", "almoço", "lunch", "jantar", "dinner", "restaurant", "restaurante")):
            return "\U0001F37D\uFE0F"
        if any(keyword in normalized_title for keyword in ("jardim", "garden", "walk", "passeio", "tejo", "river")):
            return "\U0001F33F"
        return "\U0001F3DB\uFE0F"

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if repaired_lines and repaired_lines[-1] != "":
                repaired_lines.append("")
            continue

        if stripped == "---":
            if repaired_lines and repaired_lines[-1] != "---":
                repaired_lines.append("---")
            continue

        if _SOURCE_LINE_RE.match(stripped):
            if repaired_lines and repaired_lines[-1] not in {"", "---"}:
                repaired_lines.extend(["", "---", ""])
            repaired_lines.append(stripped)
            continue

        plain = _strip_markdown_formatting(stripped)
        plain = re.sub(r"(\d{1,2})\s*:\s*(\d{2})", r"\1:\2", plain)
        plain = re.sub(r"^(?:[-*•]\s*)?#\s+", "", plain).strip()
        normalized_plain = _normalize_planner_line(stripped)

        title_candidate = itinerary_title_match(normalized_plain) or itinerary_title_match(plain)
        if not title_fixed and title_candidate:
            repaired_lines.append(f"### \U0001F4C5 {title_candidate}")
            title_fixed = True
            continue

        calendar_title_match = re.match(r"^(?:###\s+)?📅\s+(?P<title>.+)$", plain)
        if not title_fixed and calendar_title_match:
            repaired_lines.append(f"### 📅 {calendar_title_match.group('title').strip().rstrip(',:- ')}")
            title_fixed = True
            continue

        timed_match = re.match(
            r"^(?P<emoji>[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+)?\s*(?P<time>\d{1,2}:\d{2})\s*[·\-–—:]\s*(?P<title>[A-Za-zÀ-ÿ].+)$",
            normalized_plain,
        )
        if timed_match and "atualizado" not in normalized_plain.lower() and "updated" not in normalized_plain.lower():
            title = timed_match.group("title").strip(" -—–")
            if _is_planner_metadata_line(title):
                metadata_match = re.match(r"^(?P<label>[^:]{2,60})\s*:\s*(?P<content>.+)$", title)
                metadata_icon = (timed_match.group("emoji") or "📍").strip() or "📍"
                if metadata_match:
                    repaired_lines.append(
                        f"- {metadata_icon} **{metadata_match.group('label').strip()}**: {metadata_match.group('content').strip()}"
                    )
                else:
                    repaired_lines.append(f"- {metadata_icon} {title}")
                continue
            else:
                icon = timed_card_icon(title)
                repaired_lines.append(f"### {icon} {timed_match.group('time')} · {title}")
                continue

        bracketed_timed_match = re.match(
            r"^(?P<emoji>[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+)?\s*\[(?P<time>\d{1,2}:\d{2})\]\s*[\-–—:]\s*(?P<title>.+)$",
            normalized_plain,
        )
        if bracketed_timed_match:
            title = bracketed_timed_match.group("title").strip(" -—–")
            if _is_planner_metadata_line(title):
                metadata_match = re.match(r"^(?P<label>[^:]{2,60})\s*:\s*(?P<content>.+)$", title)
                metadata_icon = (bracketed_timed_match.group("emoji") or "📍").strip() or "📍"
                if metadata_match:
                    repaired_lines.append(
                        f"- {metadata_icon} **{metadata_match.group('label').strip()}**: {metadata_match.group('content').strip()}"
                    )
                else:
                    repaired_lines.append(f"- {metadata_icon} {title}")
            else:
                icon = (bracketed_timed_match.group("emoji") or "").strip() or timed_card_icon(title)
                repaired_lines.append(f"### {icon} {bracketed_timed_match.group('time')} · {title}")
            continue

        canonical_section = None
        if not re.match(r"^(?:[-*•]\s*)", stripped):
            canonical_section = canonical_planner_section(plain)
        if canonical_section:
            repaired_lines.append(canonical_section)
            continue

        if re.match(r"^(?:[-*•]\s*)", stripped):
            bullet_plain = re.sub(r"^(?:[-*•]\s*)", "", plain).strip()
            bullet_normalized = _strip_accents_compat(bullet_plain).lower()
            if any(
                token in bullet_normalized
                for token in ("notas", "dicas", "sugest", "condic", "como chegar", "desloca", "recommend", "options")
            ):
                canonical_bullet_section = canonical_planner_section(bullet_plain)
                if canonical_bullet_section:
                    repaired_lines.append(canonical_bullet_section)
                    continue

            calendar_window_match = re.search(
                r"(?P<window>\d{1,2}:\d{2}\s*(?:[–—−‑-]|to)\s*\d{1,2}:\d{2})$",
                bullet_plain,
                flags=re.IGNORECASE,
            )
            if bullet_plain.startswith("📅 ") and calendar_window_match and not title_fixed:
                bullet_label = bullet_plain[2:calendar_window_match.start()].strip().rstrip(",:- ")
                bullet_window = re.sub(
                    r"\s*(?:(?P<dash>[–—−‑-])|(?P<word>to))\s*",
                    lambda match: match.group("dash") or " to ",
                    calendar_window_match.group("window").strip(),
                    flags=re.IGNORECASE,
                )
                repaired_lines.append(f"### 📅 {bullet_label}")
                title_fixed = True
                window_label = "Janela sugerida" if is_pt else "Suggested window"
                repaired_lines.append(f"⏰ **{window_label}:** {bullet_window}")
                continue

            bullet_field_match = re.match(
                r"^(?P<icon>[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+)\s+(?P<label>[^:]{2,60})\s*:\s*(?P<content>.+)$",
                bullet_plain,
            )
            if bullet_field_match:
                bullet_icon = bullet_field_match.group("icon").strip()
                bullet_label = bullet_field_match.group("label").strip().rstrip(",")
                bullet_content = bullet_field_match.group("content").strip()
                normalized_label = _strip_accents_compat(bullet_label).lower()
                if (
                    not title_fixed
                    and bullet_icon == "📅"
                    and any(token in normalized_label for token in ("recomend", "itiner", "roteiro", "plano"))
                ):
                    repaired_lines.append(f"### 📅 {bullet_label}")
                    title_fixed = True
                    if bullet_content:
                        window_label = "Janela sugerida" if is_pt else "Suggested window"
                        repaired_lines.append(f"- ⏰ **{window_label}:** {bullet_content}")
                    continue
                repaired_lines.append(
                    f"- {bullet_icon} **{bullet_label}:** {bullet_content}"
                    if bullet_icon == "⏰" and normalized_label in {"janela sugerida", "suggested window"}
                    else f"- {bullet_icon} **{bullet_label}**: {bullet_content}"
                )
                continue

            bullet_poi_match = re.match(
                r"^(?P<icon>[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+)\s+(?P<title>[A-Za-zÀ-ÿ].+)$",
                bullet_plain,
            )
            if (
                bullet_poi_match
                and bullet_poi_match.group("icon").strip() not in section_icons
                and ":" not in bullet_poi_match.group("title")
            ):
                repaired_lines.append(
                    f"- {bullet_poi_match.group('icon').strip()} **{bullet_poi_match.group('title').strip()}**"
                )
                continue

        header_match = re.match(r"^###\s+(?P<icon>[^\s]+)\s+(?P<title>.+)$", plain)
        if header_match:
            icon = header_match.group("icon").strip()
            title = header_match.group("title").strip()
            if icon in section_icons and not _TIMED_SECTION_HEADER_RE.match(f"{icon} {title}"):
                repaired_lines.append(f"**{icon} {title}**")
                continue

        icon_heading_match = re.match(r"^(?P<icon>[^\s]+)\s+(?P<title>.+)$", plain)
        if icon_heading_match:
            icon = icon_heading_match.group("icon").strip()
            title = icon_heading_match.group("title").strip()
            if icon in section_icons and ":" not in title:
                repaired_lines.append(f"**{icon} {title}**")
                continue

        poi_heading_match = re.match(
            r"^(?P<icon>[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+)\s+(?P<title>[A-Za-zÀ-ÿ].+)$",
            plain,
        )
        if poi_heading_match:
            icon = poi_heading_match.group("icon").strip()
            title = poi_heading_match.group("title").strip()
            if icon not in section_icons and ":" not in title:
                repaired_lines.append(f"- {icon} **{title}**")
                continue

        repaired_lines.append(plain)

    deduped_lines: list[str] = []
    previous_nonempty = ""
    seen_section_headings: set[str] = set()
    for line in repaired_lines:
        stripped_line = line.strip()
        is_semantic_section = (
            stripped_line.startswith("**")
            and any(icon in stripped_line for icon in section_icons)
        )
        if (
            stripped_line
            and stripped_line == previous_nonempty
            and is_semantic_section
        ):
            continue
        if is_semantic_section and stripped_line in seen_section_headings:
            continue
        deduped_lines.append(line)
        if is_semantic_section:
            seen_section_headings.add(stripped_line)
        if stripped_line:
            previous_nonempty = stripped_line

    repaired = clean_newlines("\n".join(deduped_lines)).strip()
    planner_like_output = any(
        token in repaired
        for token in (
            "**🚇 Como Chegar e Deslocação**",
            "**📍 Sugestões para a Visita**",
            "**✨",
            "### 🏛️",
            "### 🌿",
            "### 🍽️",
            "### ☕",
            "### 🥐",
        )
    )
    if repaired and not repaired.startswith("### ") and re.search(r"\b(itiner[aáàâã]rio|itinerary|plano|roteiro)\b", _strip_accents_compat(repaired), re.IGNORECASE):
        first_line, *rest = repaired.splitlines()
        maybe_title = itinerary_title_match(first_line)
        if maybe_title:
            repaired = "\n".join([f"### \U0001F4C5 {maybe_title}", *rest]).strip()
    elif repaired and not repaired.startswith("### ") and planner_like_output:
        default_title = "### \U0001F4C5 Itinerário Sugerido" if is_pt else "### \U0001F4C5 Suggested Itinerary"
        repaired = f"{default_title}\n\n{repaired}".strip()

    previous = None
    while repaired != previous:
        previous = repaired
        repaired = re.sub(
            r"(?P<header>\*\*[^\n]+\*\*)\n\n---\n\n(?P=header)",
            r"\g<header>",
            repaired,
        )

    repaired = re.sub(
        r"(?m)^-?\s*⏰\s+\*\*(Janela\s+[Ss]ugerida|Suggested\s+[Ww]indow):\*\*\s*(.+)$",
        r"⏰ **\1:** \2",
        repaired,
    )
    repaired = re.sub(
        r"(?m)^-?\s*⏰\s+\*\*(Janela\s+[Ss]ugerida|Suggested\s+[Ww]indow)\*\*:\s*(.+)$",
        r"⏰ **\1:** \2",
        repaired,
    )
    repaired = re.sub(
        r"(?m)^-?\s*⏰\s+(Janela\s+[Ss]ugerida|Suggested\s+[Ww]indow):\s*(.+)$",
        r"⏰ **\1:** \2",
        repaired,
    )
    repaired = re.sub(
        r"(?<!\n)\s+-\s+(?=(?:🚌|🚫|✅|✨|⚠️|🔹)\s+\*\*)",
        "\n- ",
        repaired,
    )
    repaired = re.sub(
        r"(?<!\n)\s+-\s+(?=(?:🚌|🚫|✅|✨|⚠️|🔹)\s+[A-Za-zÀ-ÿ])",
        "\n- ",
        repaired,
    )
    repaired = re.sub(
        r"(?m)^(###\s+\S+\s+\d{1,2}:\d{2})\s+·\s+\d{1,2}:\d{2}\s+·\s+",
        r"\1 · ",
        repaired,
    )
    repaired = re.sub(
        r"(?m)^-\s+(?P<icon>[\U0001F300-\U0001FAFF\u2300-\u27BF\uFE0F\u200D]+)\s+\*\*(?P<hour>\d{1,2})\*\*:\s*(?P<minute>\d{2})\s+·\s+\d{1,2}:\d{2}\s+·\s+(?P<title>.+)$",
        r"### \g<icon> \g<hour>:\g<minute> · \g<title>",
        repaired,
    )
    repaired = re.sub(
        r"(?m)^-\s+🚌\s+Transporte:\s*(.+)$",
        r"- 🚌 **Transporte**: \1",
        repaired,
    )

    normalized_lines: list[str] = []
    has_window_line = False
    canonical_window_label = "Janela Sugerida" if is_pt else "Suggested Window"
    malformed_window_pattern = re.compile(
        r"^-?\s*🏛️\s+\*\*Recomenda(?:ção|ções)\s+para(?:\s+[Aa]s)?\s+(?P<hour>\d{1,2})\*\*:\s*(?P<rest>\d{2}\s*[–—−‑-]\s*\d{1,2}:\d{2})$",
        re.IGNORECASE,
    )
    for line in repaired.splitlines():
        stripped_line = line.strip()
        if re.match(r"^⏰\s+\*\*(?:Janela\s+[Ss]ugerida|Suggested\s+[Ww]indow):\*\*", stripped_line):
            has_window_line = True
            normalized_lines.append(line)
            continue

        malformed_window_match = malformed_window_pattern.match(stripped_line)
        if malformed_window_match:
            if not has_window_line:
                normalized_lines.append(
                    f"⏰ **{canonical_window_label}:** {malformed_window_match.group('hour')}:{malformed_window_match.group('rest').strip()}"
                )
                has_window_line = True
            continue

        normalized_lines.append(line)

    repaired = "\n".join(normalized_lines)
    repaired = add_section_spacing(repaired)
    repaired = clean_newlines(repaired).strip()

    return repaired


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

    if agent_name == "weather":
        finalized = format_response(finalized)
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
        researcher_kind = infer_researcher_source_kind(user_query=user_query, text=finalized)
        already_structured_event_cards = bool(
            researcher_kind == "events"
            and re.search(r"(?m)^###\s+[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s+.+$", finalized)
        )
        service_structured = structure_service_lookup_markdown(
            finalized,
            language=preferred_language,
        )
        if already_structured_event_cards:
            finalized = strip_researcher_meta_notes(finalized)
        elif service_structured != finalized:
            finalized = service_structured
        else:
            finalized = format_response(finalized)
            finalized = clean_researcher_tool_artifacts(finalized)
            finalized = structure_ranked_research_results(finalized)
            finalized = strip_researcher_meta_notes(finalized)
        if _ACCESSIBILITY_QUERY_RE.search(user_query or ""):
            finalized = strip_unconfirmed_accessibility_claims(
                finalized,
                language=preferred_language,
            )
        finalized = canonicalize_local_information_terms(finalized, language=preferred_language)
        researcher_kind = infer_researcher_source_kind(user_query=user_query, text=finalized)
        if researcher_kind == "events":
            finalized = format_researcher_event_cards(
                finalized,
                language=preferred_language,
                user_query=user_query,
            )
        elif researcher_kind != "events":
            finalized = format_researcher_card(
                finalized,
                language=preferred_language,
                user_query=user_query,
            )
        finalized = final_visual_pass(finalized)
        finalized = canonicalize_visitlisboa_source_line(
            finalized,
            user_query=user_query,
            language=preferred_language,
        )
    elif agent_name in {"planner", "transport"}:
        finalized = strip_unsupported_closing_offers(finalized)
        finalized = canonicalize_local_information_terms(finalized, language=preferred_language)
        if agent_name == "transport":
            finalized = strip_transport_weather_disclaimers(finalized)
            finalized = canonicalize_transport_terms(finalized, language=preferred_language)
            finalized = strip_technical_output_artifacts(finalized)
            finalized = structure_transport_markdown(finalized)
            finalized = soften_internal_markdown_headers(
                finalized,
                preserve_first_header=True,
                preserve_timed_cards=False,
            )
            finalized = format_response(finalized)
            finalized = canonicalize_transport_terms(finalized, language=preferred_language)
            finalized = ensure_transport_notes_heading(finalized, language=preferred_language)
            finalized = normalize_transport_notes_block(finalized)
        else:
            finalized = structure_planner_markdown(finalized)
            finalized = soften_internal_markdown_headers(
                finalized,
                preserve_first_header=True,
                preserve_timed_cards=True,
            )
            finalized = format_response(finalized)
            finalized = repair_planner_markdown_contract(finalized, language=preferred_language)
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
    cp_context = re.search(
        r'\bCP\b|Comboios de Portugal|CP Trains|\bcomboios?\s+via\b|\btrain\s+via\b',
        text,
        re.IGNORECASE,
    )

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
        r"^\*\*[⛅🚇📍✨🔎⚠️📝].*\*\*$",  # Bold semantic section headings used by planner repair
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
                "### \U0001f687 Mobilidade em Lisboa"
                if language == "pt"
                else "### \U0001f687 Lisbon Mobility"
            )

        elif agent == "researcher":
            # Keyword-based subcategorization
            event_kw = [
                "evento", "event", "concerto", "concert", "festival",
                "espetáculo", "show", "teatro", "theatre", "theater",
                "ópera", "opera", "dança", "dance", "exposição", "exhibition",
                "feira", "fair", "summit", "conference", "congress", "forum",
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
                "estacionamento", "parking", "marketplace",
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
                    "### \U0001f4cd Destaques Locais"
                    if language == "pt"
                    else "### \U0001f4cd Local Highlights"
                )

    # --- Multi-agent (without planner) - combined titles ---
    if "weather" in agents_called and "transport" in agents_called:
        return (
            "### \U0001f9ed Meteorologia e Mobilidade"
            if language == "pt"
            else "### \U0001f9ed Weather & Mobility"
        )
    elif "weather" in agents_called:
        return (
            "### \U0001f324\ufe0f Previsão Meteorológica"
            if language == "pt"
            else "### \U0001f324\ufe0f Weather Forecast"
        )
    elif "transport" in agents_called:
        return (
            "### \U0001f687 Mobilidade em Lisboa"
            if language == "pt"
            else "### \U0001f687 Lisbon Mobility"
        )
    else:
        return (
            "### \U0001f4cd Destaques Locais"
            if language == "pt"
            else "### \U0001f4cd Local Highlights"
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
    if _TRANSPORT_ROUTE_TITLE_RE.match(first_line):
        return f"{title}\n\n{text}"
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
    text = re.sub(r"\bActualizado\b", "Atualizado", text, flags=re.IGNORECASE)
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
        9. Final visual pass (phone/Google Maps linkification, bold/time repair,
           warnings-before-source reorder, stray leading enumerator strip)

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
    text = sanitize_event_title_suffixes(text)
    text = strip_internal_sections(text)
    text = clean_decorative_separators(text)
    text = normalize_headers(text)
    text = add_section_spacing(text)
    text = add_section_separators(text)
    text = clean_newlines(text)
    text = normalize_bullets(text)
    text = ensure_clickable_urls(text)
    text = final_visual_pass(text)

    return text.strip()


# ==========================================================================
# Final visual pass (Phase 2 polish)
# ==========================================================================

_PHONE_PT_RE = re.compile(r"(?<!\d)(\+?351)\s*(\d{3})\s*(\d{3})\s*(\d{3})(?!\d)")
# Collapse accidental whitespace on either side of a time-range colon so the
# bold markdown span wrapping the time doesn't break the renderer.
_BOLD_TIME_SPACE_AFTER_RE = re.compile(r"(\d{1,2}):\s+(\d{2})")
_BOLD_TIME_SPACE_BEFORE_RE = re.compile(r"(\d{1,2})\s+:(\d{2})")
_ADDRESS_LINE_RE = re.compile(
    r"(^|\n)(\s*[-*]?\s*)(📍\s*\*\*(?:Morada|Address|Location|Localiza(?:ç|c)[ãa]o|Endere[çc]o)\s*:?\s*\*\*:?\s*)(.+?)(?=\n|$)",
    re.IGNORECASE,
)
_COORDINATE_PAIR_RE = re.compile(
    r"(?P<prefix>^|\s|\()(?P<lat>-?\d{1,2}\.\d+)\s*,\s*(?P<lon>-?\d{1,3}\.\d+)(?P<suffix>\)|\s|$)"
)


def _gmaps_link(address: str) -> str:
    """Return a Google Maps search URL with the address URL-encoded."""
    from urllib.parse import quote_plus

    clean = address.strip().rstrip(",.;:")
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(clean)}"


def _gmaps_coordinate_link(lat: str, lon: str) -> str:
    """Return a Google Maps search URL for a latitude/longitude pair."""
    return f"https://www.google.com/maps/search/?api=1&query={lat}%2C{lon}"


def linkify_phone_numbers(text: str) -> str:
    """Replace bare +351 phone numbers with ``tel:`` markdown links.

    Matches patterns like ``+351 213 613 000``, ``351213613000``, or
    ``213 613 000`` inside the +351 prefix and rewrites them as
    ``[+351 XXX XXX XXX](tel:+351XXXXXXXXX)`` so Streamlit renders them as
    clickable dial links.
    """
    if not text or "351" not in text:
        return text

    def _sub(match: re.Match) -> str:
        g1, g2, g3 = match.group(2), match.group(3), match.group(4)
        digits = f"{g1}{g2}{g3}"
        visible = f"+351 {g1} {g2} {g3}"
        # Skip if already inside an existing markdown link.
        start = match.start()
        window = text[max(0, start - 40):start]
        if "[" in window and "](tel:" in text[start:start + 120]:
            return match.group(0)
        # Skip if the match is already inside a ``tel:`` URL body
        # (e.g. the digits portion of ``](tel:+351213500115)``).
        prefix = text[max(0, start - 6):start]
        if "tel:" in prefix:
            return match.group(0)
        return f"[{visible}](tel:+351{digits})"

    return _PHONE_PT_RE.sub(_sub, text)


def linkify_address_lines(text: str) -> str:
    """Wrap bullet-list address values in a Google Maps link.

    Applies to lines of the form ``📍 **Address**: <value>`` (EN) or
    ``📍 **Morada**: <value>`` (PT). The link targets the Google Maps search
    endpoint. Already-linked values are left alone.
    """
    if "📍" not in text and not _COORDINATE_PAIR_RE.search(text):
        return text

    def _sub(match: re.Match) -> str:
        lead, bullet_prefix, label, value = match.group(1), match.group(2), match.group(3), match.group(4)
        stripped_value = value.strip()
        if not stripped_value or stripped_value.startswith("[") or "](" in stripped_value:
            return match.group(0)
        link = _gmaps_link(stripped_value)
        return f"{lead}{bullet_prefix}{label}[{stripped_value}]({link})"

    text = _ADDRESS_LINE_RE.sub(_sub, text)

    def _coord_sub(match: re.Match) -> str:
        prefix = match.group("prefix")
        lat = match.group("lat")
        lon = match.group("lon")
        suffix = match.group("suffix")
        raw = f"{lat}, {lon}"
        start = match.start()
        window = text[max(0, start - 16):start + 16]
        if "](" in window:
            return match.group(0)
        return f"{prefix}[{raw}]({_gmaps_coordinate_link(lat, lon)}){suffix}"

    return _COORDINATE_PAIR_RE.sub(_coord_sub, text)


def repair_bold_time_spacing(text: str) -> str:
    """Collapse accidental spaces inside time ranges such as ``19: 00`` -> ``19:00``.

    This avoids breaking markdown bold spans that wrap time ranges, where an
    inner ``:<space>`` fragment was causing the renderer to close the bold
    prematurely (Q20 regression).
    """
    if not text or ":" not in text:
        return text
    text = _BOLD_TIME_SPACE_AFTER_RE.sub(r"\1:\2", text)
    text = _BOLD_TIME_SPACE_BEFORE_RE.sub(r"\1:\2", text)
    return text


def strip_stray_leading_enumerator(text: str) -> str:
    """Remove a stray ``1.`` when it is the only numeric marker inside a card.

    The LLM sometimes emits ``### Card title\\n1. **Name**`` for a single
    entry which renders as a half-broken ordered list. We strip the ``1.``
    when no ``2.`` follows within the same card block.
    """
    if not text:
        return text
    lines = text.splitlines()
    out: List[str] = []
    inside_card = False
    card_enum_line_idx: Optional[int] = None
    card_has_followup = False

    def _flush_card() -> None:
        nonlocal card_enum_line_idx, card_has_followup
        if card_enum_line_idx is not None and not card_has_followup:
            prev = out[card_enum_line_idx]
            # Handle both ``1. content`` and a bare orphan ``1.`` line.
            if re.match(r"^\s*1\.\s*$", prev):
                out[card_enum_line_idx] = re.sub(r"^(\s*)1\.\s*$", r"\1", prev)
            else:
                out[card_enum_line_idx] = re.sub(r"^(\s*)1\.\s+", r"\1", prev)
        card_enum_line_idx = None
        card_has_followup = False

    for line in lines:
        if line.lstrip().startswith("### "):
            _flush_card()
            inside_card = True
            out.append(line)
            continue
        if inside_card:
            if card_enum_line_idx is None and re.match(r"^\s*1\.\s*(?:$|\S)", line):
                card_enum_line_idx = len(out)
            elif card_enum_line_idx is not None and re.match(r"^\s*[2-9]\.\s+", line):
                card_has_followup = True
        out.append(line)

    _flush_card()
    return "\n".join(out)


def strip_orphan_bold_markers(text: str) -> str:
    """Remove standalone or dangling bold markers that are not part of a valid pair."""
    if not text or "**" not in text:
        return text
    text = re.sub(r"(?m)^\s*\*\*\s*$", "", text)
    if text.count("**") % 2 == 1:
        text = re.sub(r"\*\*(?!.*\*\*)", "", text, count=1, flags=re.DOTALL)
    return clean_newlines(text).rstrip("\n")


def ensure_blank_lines_before_emoji_fields(text: str) -> str:
    """Insert a blank line before dense emoji-prefixed field lines when needed."""
    if not text:
        return text
    field_prefixes = ("📍", "📅", "⏱️", "📞", "🌐", "⭐", "💶", "💰", "🎟️", "📝", "📂", "🕐", "🗺️", "📏")
    lines = text.splitlines()
    output_lines: list[str] = []

    for line in lines:
        stripped = line.lstrip()
        if line == stripped and stripped.startswith(field_prefixes) and output_lines:
            previous_line = output_lines[-1].strip()
            if previous_line and not output_lines[-1].startswith(("### ", "#### ")):
                output_lines.append("")
        output_lines.append(line)

    return "\n".join(output_lines)


def reorder_warnings_before_source(text: str) -> str:
    """Move ``⚠️`` warning lines that appear AFTER the final source footer
    to immediately before the footer (Q3 regression).
    """
    return _reorder_marker_before_source(text, marker="⚠️")


def reorder_tips_before_source(text: str) -> str:
    """Move ``💡`` tip lines that appear AFTER the final source footer back
    to immediately before the footer. Same shape as
    :func:`reorder_warnings_before_source`, applied to the tip marker.
    """
    return _reorder_marker_before_source(text, marker="💡")


def repair_known_live_typos(text: str) -> str:
    """Clean a small set of recurring QA/LLM typo artefacts from final output.

    These are not semantic rewrites. They fix repeated-letter glitches observed
    in live runs after the final repair pass, such as ``iis`` or ``orrigin``.
    The replacements are intentionally narrow and word-bounded so they do not
    affect normal prose.
    """
    if not text:
        return text

    replacements = [
        (r"\bTTour\b", "Tour"),
        (r"\biis\b", "is"),
        (r"\borrigin\b", "origin"),
        (r"\bveryy\b", "very"),
        (r"\bmu+ito\b", "muito"),
        (r"\bG+TFS\b", "GTFS"),
        (r"\bMetropolitanaa\b", "Metropolitana"),
        (r"\bestadoo\b", "estado"),
        (r"\béé\b", "é"),
        (r"\bfo+ntes\b", "fontes"),
        (r"\bope+racional\b", "operacional"),
        (r"\bexpl[ií]cit+a\b", "explícita"),
        (r"\boficiai+s\b", "oficiais"),
        (r"\bppara\b", "para"),
        (r"\bveri+ficar\b", "verificar"),
        (r"\bveri+ficado\b", "verificado"),
        (r"\breflecct\b", "reflect"),
        (r"\bafffected\b", "affected"),
        (r"\bcoerennte\b", "coerente"),
        (r"\bconfirmadoss\b", "confirmados"),
        (r"\bdependdente\b", "dependente"),
        (r"\benttre\b", "entre"),
        (r"\bEventtos\b", "Eventos"),
        (r"\brecenttes\b", "recentes"),
        (r"\balteera[cç][õo]es\b", "alterações"),
        (r"\brefletiir\b", "refletir"),
        (r"\brresposta\b", "resposta"),
        (r"\btradiciional\b", "tradicional"),
        (r"\bttradicional\b", "tradicional"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def ensure_transport_notes_heading(text: str, language: str = "en") -> str:
    """Insert a transport notes heading when a disclaimer block follows a separator."""
    if not text:
        return text

    heading = "### ⚠️ Notas Úteis" if language == "pt" else "### ⚠️ Helpful Notes"
    if heading in text:
        return text

    return re.sub(
        r"(\n---\n\n)(?=-\s*⚠️)",
        rf"\1{heading}\n\n",
        text,
        count=1,
    )


def normalize_transport_notes_block(text: str) -> str:
    """Render transport note warnings as plain paragraphs instead of markdown bullets."""
    if not text or ("Notas Úteis" not in text and "Helpful Notes" not in text):
        return text

    lines = text.splitlines()
    normalized_lines: list[str] = []
    inside_notes = False

    for line in lines:
        stripped = line.strip()
        if stripped in {"### ⚠️ Notas Úteis", "### ⚠️ Helpful Notes"}:
            inside_notes = True
            normalized_lines.append(line)
            continue

        if inside_notes:
            if _SOURCE_LINE_RE.match(stripped) or stripped.startswith("### "):
                inside_notes = False
                normalized_lines.append(line)
                continue

            bullet_match = re.match(r"^\s*[-*]\s*(⚠️\s*.+)$", stripped)
            if bullet_match:
                normalized_lines.append(bullet_match.group(1))
                continue

        normalized_lines.append(line)

    return "\n".join(normalized_lines)


def _reorder_marker_before_source(text: str, marker: str) -> str:
    """Shared helper: move any line containing ``marker`` and appearing AFTER the
    source footer back to just before the footer.
    """
    if not text or "📌" not in text or marker not in text:
        return text

    source_re = re.compile(r"(?m)^(📌\s*\*\*(?:Fonte|Source):\*\*.*)$")
    source_match = source_re.search(text)
    if not source_match:
        return text

    before = text[:source_match.start()]
    source_line = source_match.group(1)
    after = text[source_match.end():]

    marker_escaped = re.escape(marker)
    line_re = re.compile(r"(?m)^(?:\s*[-*]\s*)?" + marker_escaped + r"[^\n]*")
    hits = line_re.findall(after)
    if not hits:
        return text

    remaining_after = line_re.sub("", after)
    remaining_after = re.sub(r"\n{3,}", "\n\n", remaining_after).strip("\n")

    block_lines: list[str] = []
    for line in hits:
        stripped = line.strip()
        if stripped.startswith(("-", "*")):
            block_lines.append(line)
        else:
            block_lines.append(f"- {stripped}")
    block = "\n".join(block_lines)
    rebuilt = before.rstrip() + "\n\n" + block + "\n\n" + source_line
    if remaining_after.strip():
        rebuilt += "\n\n" + remaining_after.strip()
    return rebuilt


def final_visual_pass(text: str) -> str:
    """Apply the final set of visual and consistency repairs in order.

    The pass is idempotent by construction: every sub-step checks for prior
    formatting before rewriting, so running this multiple times on the same
    text returns the same output.
    """
    if not text or not isinstance(text, str):
        return text or ""
    text = repair_bold_time_spacing(text)
    text = strip_orphan_bold_markers(text)
    text = linkify_phone_numbers(text)
    text = linkify_address_lines(text)
    text = strip_stray_leading_enumerator(text)
    text = ensure_blank_lines_before_emoji_fields(text)
    text = reorder_warnings_before_source(text)
    text = reorder_tips_before_source(text)
    text = repair_known_live_typos(text)
    text = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", text)
    # Collapse triple blank lines that may have been reintroduced.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ==========================================================================
# Language fidelity (PT ↔ EN deterministic label repair)
# ==========================================================================

# Paired label translations used by `enforce_language_labels` when a response
# is meant to be entirely in one language but a worker emitted a label in the
# other. Keys are case-insensitive exact labels; the paired tuple is
# (pt_form, en_form). Only apply when the form is a *label*, not a content
# word that could cause false positives in running prose.
_LABEL_TRANSLATIONS: List[tuple] = [
    # (pt_label, en_label, is_bold_label)
    ("Categoria", "Category", True),
    ("Avaliações", "Reviews", True),
    ("Descrição", "Description", True),
    ("Morada", "Address", True),
    ("Localização", "Location", True),
    ("Endereço", "Address", True),
    ("Horário", "Hours", True),
    ("Horário de funcionamento", "Opening hours", True),
    ("Contacto", "Contact", True),
    ("Telefone", "Phone", True),
    ("Sítio", "Website", True),
    ("Website", "Website", True),
    ("Preço", "Price", True),
    ("Bilhetes", "Tickets", True),
    ("Próximo", "Next", True),
    ("Amanhã", "Tomorrow", True),
    ("Hoje", "Today", True),
    ("Fechado", "Closed", True),
    ("Aberto", "Open", True),
    ("Fonte", "Source", True),
    ("Atualizado", "Updated", True),
    ("Janela de resultados", "Results window", True),
    ("Dica", "Tip", True),
    ("Dica rápida", "Quick tip", True),
    ("Nota", "Note", True),
    ("Aviso", "Warning", True),
    ("Duração", "Duration", True),
    ("Data", "Date", True),
    ("Local", "Venue", True),
    ("Horários", "Schedule", True),
]


def _label_replace(text: str, src_label: str, dst_label: str) -> str:
    """Replace ``**src_label**`` (with optional colon) with the target form.

    Matches only bolded labels and keeps the trailing colon, emoji, and the
    rest of the line untouched. This stays well inside "label repair" territory
    and never touches free-running text. Two forms are covered:

    - ``**Morada**`` (no trailing colon inside bold)
    - ``**Morada:**`` (trailing colon INSIDE the bold, emitted by many LLMs)

    In both cases the colon is preserved on the output side when it was present.
    """
    if not text or not src_label or src_label.lower() == dst_label.lower():
        return text
    # Form A: **Label:**   (colon inside the bold)
    pattern_with_colon = re.compile(
        r"(?<!\w)\*\*" + re.escape(src_label) + r"\s*:\s*\*\*",
        re.IGNORECASE,
    )
    text = pattern_with_colon.sub(f"**{dst_label}:**", text)
    # Form B: **Label**    (no colon, any trailing punctuation stays untouched)
    pattern_plain = re.compile(
        r"(?<!\w)\*\*" + re.escape(src_label) + r"\*\*",
        re.IGNORECASE,
    )
    text = pattern_plain.sub(f"**{dst_label}**", text)
    return text


def enforce_language_labels(text: str, language: str) -> str:
    """Rewrite well-known PT/EN label pairs so the response stays in one language.

    This is a deterministic safety net for the case where an LLM answer is in
    English but carries one or two Portuguese labels inherited from a worker
    output (or vice versa). It only rewrites *bolded* labels (``**Label**``)
    so it cannot damage prose. The helper is a no-op when ``language`` is not
    ``"pt"`` or ``"en"``.
    """
    if not text or not isinstance(text, str):
        return text or ""
    normalized = language if language in {"pt", "en"} else None
    if normalized is None:
        return text

    for pt_label, en_label, _is_bold in _LABEL_TRANSLATIONS:
        if normalized == "en":
            text = _label_replace(text, pt_label, en_label)
        else:
            text = _label_replace(text, en_label, pt_label)
    return text


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
