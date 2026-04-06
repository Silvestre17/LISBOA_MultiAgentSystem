# ==========================================================================
# Master Thesis - VisitLisboa Semantic Search Tools
#   - André Filipe Gomes Silvestre, 20240502
#
#   Semantic search tools for VisitLisboa data using ChromaDB vector store.
#   Features:
#     - Semantic search over events using embeddings
#     - Semantic search over places/attractions using embeddings
#     - Category filtering with semantic understanding
#     - DATE FILTERING for events (critical feature)
#     - Fallback to JSON when vector store unavailable
#
#   Data Sources:
#     - lisbon_events collection: Cultural events, exhibitions, festivals
#     - lisbon_places collection: Museums, monuments, restaurants
#
#   Note: Requires vector store to be built first with vector_store.py
# ==========================================================================

# Required libraries:
# pip install langchain-core langchain-chroma langchain-huggingface

import json
import logging
import math
import os
import re
import unicodedata
import warnings
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

# Suppress chromadb telemetry warnings
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

from langchain_core.tools import tool

try:
    from config import Config
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import Config

logger = logging.getLogger(__name__)

# Collection names (must match vector_store.py)
COLLECTION_PDF = "lisbon_pdf"
COLLECTION_PLACES = "lisbon_places"
COLLECTION_EVENTS = "lisbon_events"


# ==========================================================================
# Date Parsing Utilities
# ==========================================================================

def parse_date_range(date_query: Optional[str]) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Parses natural language date queries into date range.

    Args:
        date_query: Natural language like 'today', 'tomorrow', 'next week',
                   'this weekend', 'January', '2025-01-15', etc.

    Returns:
        Tuple of (start_date, end_date) or (None, None) if no date filter.
    """
    if not date_query:
        return None, None

    date_query = date_query.lower().strip()
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Handle specific keywords
    if date_query in ['today', 'hoje']:
        return today, today + timedelta(days=1)

    elif date_query in ['tomorrow', 'amanhã', 'amanha']:
        tomorrow = today + timedelta(days=1)
        return tomorrow, tomorrow + timedelta(days=1)

    elif date_query in ['this week', 'esta semana', 'week', 'semana']:
        # Start from today, end at next Sunday
        days_until_sunday = 6 - today.weekday()
        if days_until_sunday < 0:
            days_until_sunday = 0
        end = today + timedelta(days=days_until_sunday + 1)
        return today, end

    elif date_query in ['next week', 'próxima semana', 'proxima semana']:
        # Next Monday to next Sunday
        days_until_monday = (7 - today.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        start = today + timedelta(days=days_until_monday)
        end = start + timedelta(days=7)
        return start, end

    elif date_query in ['this weekend', 'este fim de semana', 'fim de semana', 'weekend']:
        # Saturday and Sunday of this week
        days_until_saturday = (5 - today.weekday()) % 7
        if days_until_saturday == 0 and today.weekday() == 5:
            # It's Saturday
            start = today
        elif today.weekday() == 6:
            # It's Sunday
            start = today
        else:
            start = today + timedelta(days=days_until_saturday)
        end = start + timedelta(days=2)  # Through Sunday
        return start, end

    elif date_query in ['this month', 'este mês', 'este mes']:
        start = today.replace(day=1)
        # Last day of month
        if today.month == 12:
            end = today.replace(year=today.year + 1, month=1, day=1)
        else:
            end = today.replace(month=today.month + 1, day=1)
        return start, end

    elif date_query in ['next month', 'próximo mês', 'proximo mes']:
        if today.month == 12:
            start = today.replace(year=today.year + 1, month=1, day=1)
            end = today.replace(year=today.year + 1, month=2, day=1)
        else:
            start = today.replace(month=today.month + 1, day=1)
            if today.month + 1 == 12:
                end = today.replace(year=today.year + 1, month=1, day=1)
            else:
                end = today.replace(month=today.month + 2, day=1)
        return start, end

    elif date_query in ['this year', 'este ano', 'year', 'ano']:
        start = today.replace(month=1, day=1)
        end = today.replace(year=today.year + 1, month=1, day=1)
        return start, end

    elif date_query in ['next year', 'próximo ano', 'proximo ano']:
        start = today.replace(year=today.year + 1, month=1, day=1)
        end = today.replace(year=today.year + 2, month=1, day=1)
        return start, end

    # Handle month names
    month_names = {
        'january': 1, 'janeiro': 1, 'jan': 1,
        'february': 2, 'fevereiro': 2, 'feb': 2, 'fev': 2,
        'march': 3, 'março': 3, 'marco': 3, 'mar': 3,
        'april': 4, 'abril': 4, 'apr': 4, 'abr': 4,
        'may': 5, 'maio': 5, 'mai': 5,
        'june': 6, 'junho': 6, 'jun': 6,
        'july': 7, 'julho': 7, 'jul': 7,
        'august': 8, 'agosto': 8, 'aug': 8, 'ago': 8,
        'september': 9, 'setembro': 9, 'sep': 9, 'set': 9,
        'october': 10, 'outubro': 10, 'oct': 10, 'out': 10,
        'november': 11, 'novembro': 11, 'nov': 11,
        'december': 12, 'dezembro': 12, 'dec': 12, 'dez': 12,
    }

    for month_name, month_num in month_names.items():
        if month_name in date_query:
            year = today.year
            # If month already passed, assume next year
            if month_num < today.month:
                year += 1
            start = datetime(year, month_num, 1)
            if month_num == 12:
                end = datetime(year + 1, 1, 1)
            else:
                end = datetime(year, month_num + 1, 1)
            return start, end

    # Try to parse ISO date (YYYY-MM-DD)
    try:
        date = datetime.strptime(date_query, '%Y-%m-%d')
        return date, date + timedelta(days=1)
    except ValueError:
        pass

    # Try DD/MM/YYYY
    try:
        date = datetime.strptime(date_query, '%d/%m/%Y')
        return date, date + timedelta(days=1)
    except ValueError:
        pass

    if re.fullmatch(r'(?:19|20)\d{2}', date_query):
        year = int(date_query)
        return datetime(year, 1, 1), datetime(year + 1, 1, 1)

    # Default: next 30 days (reasonable default for "upcoming" events)
    if any(word in date_query for word in ['upcoming', 'próximos', 'proximos', 'soon', 'breve']):
        return today, today + timedelta(days=30)

    return None, None


def get_event_dates(event: Dict) -> List[datetime]:
    """
    Extracts all dates from an event.

    Args:
        event: Event dictionary with 'dates' field.

    Returns:
        List of datetime objects for this event.
    """
    dates = []
    event_dates = event.get('dates', [])

    for date_entry in event_dates:
        date_info = date_entry.get('date', {})
        iso_date = date_info.get('datetime_iso')

        if iso_date:
            try:
                dt = datetime.strptime(iso_date, '%Y-%m-%d')
                dates.append(dt)
            except ValueError:
                continue

        # Handle date ranges
        if date_entry.get('type') == 'range':
            start_info = date_entry.get('start', {})
            end_info = date_entry.get('end', {})

            start_iso = start_info.get('datetime_iso')
            end_iso = end_info.get('datetime_iso')

            if start_iso:
                try:
                    dates.append(datetime.strptime(start_iso, '%Y-%m-%d'))
                except ValueError:
                    pass
            if end_iso:
                try:
                    dates.append(datetime.strptime(end_iso, '%Y-%m-%d'))
                except ValueError:
                    pass

    return dates


def filter_events_by_date(
    events: List[Dict],
    start_date: Optional[datetime],
    end_date: Optional[datetime]
) -> List[Dict]:
    """
    Filters events by date range.

    Args:
        events: List of event dictionaries.
        start_date: Start of date range (inclusive).
        end_date: End of date range (exclusive).

    Returns:
        Filtered list of events.
    """
    if not start_date and not end_date:
        return events

    filtered = []
    for event in events:
        event_dates = get_event_dates(event)

        if not event_dates:
            continue  # Skip events without dates

        # Check if any event date falls within range
        for dt in event_dates:
            in_range = True
            if start_date and dt < start_date:
                in_range = False
            if end_date and dt >= end_date:
                in_range = False

            if in_range:
                filtered.append(event)
                break  # Don't add same event multiple times

    # Sort by earliest date
    def get_earliest_date(e):
        dates = get_event_dates(e)
        return min(dates) if dates else datetime.max

    filtered.sort(key=get_earliest_date)

    return filtered


def get_event_duration_days(event: Dict) -> int:
    """
    Calculates the total duration span of an event in days.

    Single-day events return 1.
    Multi-day events return the span between first and last date.
    Long-running exhibitions (>30 days) are penalized in ranking.

    Args:
        event: Event dictionary with 'dates' field.

    Returns:
        int: Duration in days (1 for single-day, span for ranges).
    """
    dates = get_event_dates(event)
    if not dates:
        return 365  # No dates = low priority (assume permanent)

    if len(dates) == 1:
        return 1  # Single-day event

    # Calculate span between min and max dates
    min_date = min(dates)
    max_date = max(dates)
    duration = (max_date - min_date).days + 1

    return max(1, duration)


def calculate_temporal_relevance_score(
    event: Dict,
    query_start: Optional[datetime],
    query_end: Optional[datetime]
) -> float:
    """
    Calculates a temporal relevance score for ranking events.

    CRITICAL: Prioritizes ephemeral (short-duration) events over long exhibitions.
    A tourist asking "what to do today?" wants unique events, not 4-month exhibitions.

    Scoring factors:
        1. Duration penalty: Longer events get lower scores
        2. Proximity bonus: Events closer to query start date get higher scores
        3. Uniqueness bonus: Single-day events get priority

    Args:
        event: Event dictionary.
        query_start: Start of the query date range.
        query_end: End of the query date range.

    Returns:
        float: Score (higher = more relevant). Range: 0.0 to 100.0
    """
    score = 50.0  # Base score

    duration = get_event_duration_days(event)
    event_dates = get_event_dates(event)

    if not event_dates:
        return 10.0  # No dates = very low priority

    # 1. Duration penalty: Short events are more valuable for tourists
    #    Single day: +30, 2-3 days: +20, 1 week: +10, 1 month: 0, >3 months: -20
    if duration == 1:
        score += 30.0  # Single-day concert/performance
    elif duration <= 3:
        score += 20.0  # Weekend event
    elif duration <= 7:
        score += 10.0  # Week-long event
    elif duration <= 30:
        score += 0.0   # Month-long (neutral)
    elif duration <= 90:
        score -= 10.0  # Quarter-long exhibition
    else:
        score -= 20.0  # Permanent/long-running

    # 2. Proximity bonus: Events starting soon are more urgent
    if query_start:
        earliest_date = min(event_dates)
        days_until = (earliest_date - query_start).days

        if days_until <= 0:  # Today or past (but still active)
            score += 5.0
        elif days_until <= 1:  # Tomorrow
            score += 10.0
        elif days_until <= 7:  # This week
            score += 5.0
        elif days_until > 30:
            score -= 5.0  # Far in the future

    # 3. Category bonus: Certain categories are more "event-like"
    category = event.get('category', '').lower()
    ephemeral_categories = ['music', 'theater', 'concerts', 'festivals', 'sports']
    if any(cat in category for cat in ephemeral_categories):
        score += 5.0

    # Ensure score is within bounds
    return max(0.0, min(100.0, score))


def format_event_dates(event: Dict, language: str = "en") -> str:
    """Formats event dates for display."""
    dates = event.get('dates', [])
    if not dates:
        return "Data a confirmar" if language == "pt" else "Date TBA"

    formatted = []
    for date_entry in dates[:3]:  # Show max 3 dates
        if date_entry.get('type') == 'single':
            date_info = date_entry.get('date', {})
            display = date_info.get('display_text', '')
            time = date_info.get('time', '')
            if display:
                if time:
                    connector = "às" if language == "pt" else "at"
                    formatted.append(f"{display} {connector} {time}")
                else:
                    formatted.append(display)
        elif date_entry.get('type') == 'range':
            start = date_entry.get('start', {}).get('display_text', '')
            end = date_entry.get('end', {}).get('display_text', '')
            if start and end:
                connector = "a" if language == "pt" else "to"
                formatted.append(f"{start} {connector} {end}")

    if len(dates) > 3:
        if language == "pt":
            formatted.append(f"(+{len(dates) - 3} datas)")
        else:
            formatted.append(f"(+{len(dates) - 3} more dates)")

    if formatted:
        return " | ".join(formatted)
    return "Data a confirmar" if language == "pt" else "Date TBA"


def _infer_visitlisboa_output_language(query: Optional[str], language: Optional[str] = None) -> str:
    """Infers whether the tool should render PT-PT or English output."""
    if language in {"pt", "en"}:
        return language

    query_lower = (query or "").lower()
    pt_markers = [
        "que ", "quais", "esta semana", "hoje", "amanhã", "amanha",
        "próxima semana", "proxima semana", "concerto", "concertos",
        "eventos", "música", "musica", "teatro", "exposição", "exposicao",
    ]
    if any(marker in query_lower for marker in pt_markers) or re.search(r"[ãõáéíóúç]", query_lower):
        return "pt"
    return "en"


def _humanize_visitlisboa_slug(url: str) -> str:
    """Converts a VisitLisboa URL slug into a cleaner user-facing title."""
    slug = (url or "").rstrip("/").split("/")[-1]
    slug = slug.replace("_", " ").replace("-", " ")
    slug = re.sub(r"\s+", " ", slug).strip()

    numeric_suffix_match = re.match(r"^(.*?)(?:\s+(0\d{2,3}|\d{2,4}))$", slug)
    if numeric_suffix_match and len(numeric_suffix_match.group(1).split()) >= 2:
        slug = numeric_suffix_match.group(1).strip()

    if not slug:
        return ""

    normalized = slug.title()
    normalized = re.sub(r"\bDe\b", "de", normalized)
    normalized = re.sub(r"\bDa\b", "da", normalized)
    normalized = re.sub(r"\bDo\b", "do", normalized)
    normalized = re.sub(r"\bDos\b", "dos", normalized)
    return normalized.strip()


def _clean_event_title(title: Optional[str], url: str = "") -> str:
    """Returns a clean event title without technical slug suffixes."""
    raw_title = (title or "").strip()
    if raw_title and raw_title.lower() not in {"unknown event", "unknown", "n/a"}:
        cleaned = re.sub(r"\s+", " ", raw_title).strip()
        suffix_match = re.match(r"^(.*?)(?:\s+(0\d{2,3}|\d{2,4}))$", cleaned)
        if suffix_match and len(suffix_match.group(1).split()) >= 2:
            cleaned = suffix_match.group(1).strip()
        return cleaned

    humanized = _humanize_visitlisboa_slug(url)
    if humanized:
        return humanized

    return "Untitled event"


def _localize_event_price(price: Optional[str], language: str = "en") -> str:
    """Localizes common VisitLisboa price snippets."""
    raw = (price or "").strip()
    if not raw:
        return ""

    if language == "pt":
        localized = re.sub(r"\bFrom\s+(€?\d+(?:[\.,]\d+)?)\s+to\s+(€?\d+(?:[\.,]\d+)?)\b", r"de \1 a \2", raw, flags=re.IGNORECASE)
        localized = re.sub(r"\bFrom\s+", "desde ", localized, flags=re.IGNORECASE)
        localized = re.sub(r"\bFree Entry\b", "Entrada gratuita", localized, flags=re.IGNORECASE)
        localized = re.sub(r"\bPaid\b", "Pago", localized, flags=re.IGNORECASE)
        return localized

    return raw


def _localize_event_category(category: Optional[str], language: str = "en") -> str:
    """Localizes common VisitLisboa event categories."""
    raw = (category or "").strip()
    if not raw:
        return ""
    if language != "pt":
        return raw

    mapping = {
        "Main Events": "Principais eventos",
        "Exhibitions": "Exposições",
        "Music": "Música",
        "Theater Opera & Dance": "Teatro, Ópera e Dança",
        "Cinema": "Cinema",
        "Sports": "Desporto",
        "Fairs": "Feiras",
        "Festivals": "Festivais",
        "Gastronomy": "Gastronomia",
        "Others": "Outros",
        "General": "Geral",
    }
    return mapping.get(raw, raw)


def _localize_event_date_filter(date_filter: Optional[str], language: str = "en") -> str:
    """Localizes common date-filter labels for user-facing summaries."""
    raw = (date_filter or "all available dates").strip()
    if language != "pt":
        return raw

    mapping = {
        "all available dates": "todas as datas disponíveis",
        "today": "hoje",
        "tomorrow": "amanhã",
        "this week": "esta semana",
        "next week": "próxima semana",
        "this weekend": "este fim de semana",
        "this month": "este mês",
        "next month": "próximo mês",
        "this year": "este ano",
        "next year": "próximo ano",
        "upcoming": "próximos dias",
    }
    return mapping.get(raw.lower(), raw)


def _compress_event_sentence(sentence: str, max_chars: int = 210) -> str:
    """Compresses very long event sentences into a concise summary."""
    cleaned = re.sub(r"\s+", " ", sentence).strip()
    if not cleaned:
        return ""

    cleaned = re.sub(r"^With\s+[^,]+,\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r",\s*having played various styles.*$",
        ", with a long career spanning multiple styles.",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r",\s*later crossing over.*$",
        ", spanning several musical styles.",
        cleaned,
        flags=re.IGNORECASE,
    )

    if len(cleaned) <= max_chars:
        return cleaned

    split_candidates = [
        cleaned.rfind(",", 0, max_chars),
        cleaned.rfind(";", 0, max_chars),
        cleaned.rfind(":", 0, max_chars),
    ]
    split_at = max(split_candidates)
    if split_at >= 80:
        trimmed = cleaned[:split_at].strip(" ,;:")
        if trimmed.endswith(('.', '!', '?')):
            return trimmed
        return trimmed + "."

    word_trimmed = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    if word_trimmed:
        return word_trimmed + "…"
    return cleaned[:max_chars].strip() + "…"


def _summarize_event_description(text: Optional[str], max_chars: int = 210) -> str:
    """Builds a concise user-facing event description from long raw source text."""
    if not text:
        return ""

    meaningful_lines: List[str] = []
    for raw_line in str(text).splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        lower_line = line.lower()
        if lower_line.startswith((
            "dates and schedules", "tickets:", "ticket:", "all at ",
            "(metro)", "(bus)", "(tram)", "(train)", "(ferry)",
            "wednesday", "thursday", "friday", "saturday", "sunday",
            "monday", "tuesday", "until ", "from ", "daily", "free entry",
            "free admission", "paid", "tickets available", "more info",
        )):
            continue
        if re.match(r"^(?:www\.|https?://)", lower_line):
            continue
        meaningful_lines.append(line)

    cleaned_text = " ".join(meaningful_lines)
    cleaned_text = re.sub(r"\[[^\]]+\]", " ", cleaned_text)
    cleaned_text = re.sub(r"https?://\S+", " ", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    if not cleaned_text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return _compress_event_sentence(cleaned_text, max_chars=max_chars)

    selected = _compress_event_sentence(sentences[0], max_chars=max_chars)
    if len(selected) < 110 and len(sentences) > 1:
        candidate = f"{selected} {_compress_event_sentence(sentences[1], max_chars=max_chars)}".strip()
        if len(candidate) <= max_chars + 30:
            selected = candidate

    return selected.strip()


def _format_event_duration_label(duration: int, language: str = "en") -> str:
    """Formats event duration in a user-friendly way."""
    if language == "pt":
        if duration == 1:
            return "🎯 Um só dia"
        if duration <= 3:
            return f"📆 {duration} dias"
        if duration <= 7:
            return "📅 Cerca de 1 semana"
        if duration <= 30:
            return "📅 Cerca de 1 mês"
        return f"🏛️ Longa duração ({duration} dias)"

    if duration == 1:
        return "🎯 Single day"
    if duration <= 3:
        return f"📆 {duration} days"
    if duration <= 7:
        return "📅 About 1 week"
    if duration <= 30:
        return "📅 About 1 month"
    return f"🏛️ Long-running ({duration} days)"


def _format_event_filter_summary(
    query: Optional[str],
    category: Optional[str],
    date_filter: Optional[str],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    total_results: int,
    shown_results: int,
    offset: int = 0,
    language: str = "en",
) -> List[str]:
    """Builds a contextual summary for the event filter and result count."""
    normalized_filter = _localize_event_date_filter(date_filter, language=language)
    if start_date and end_date:
        connector = "a" if language == "pt" else "to"
        date_window = f"{start_date.strftime('%Y-%m-%d')} {connector} {(end_date - timedelta(days=1)).strftime('%Y-%m-%d')}"
    elif start_date:
        date_window = start_date.strftime('%Y-%m-%d')
    else:
        date_window = "intervalo aberto" if language == "pt" else "open range"

    normalized_query = (query or "").strip()
    normalized_category = (category or "").strip()
    normalized_category = _localize_event_category(normalized_category, language=language)
    shown_from = offset + 1 if shown_results > 0 else 0
    shown_to = offset + shown_results if shown_results > 0 else 0

    if language == "pt":
        scope_parts = [normalized_filter if not date_filter else f"{normalized_filter} ({date_window})"]
        scope_parts.append(normalized_category if normalized_category else "todas as categorias")
        if normalized_query:
            scope_parts.append(f"foco temático: {normalized_query}")
        else:
            scope_parts.append("pesquisa geral de eventos")
        return [
            "- 🧾 **Resumo da pesquisa**",
            f"    - 🧭 **Filtro aplicado:** {', '.join(scope_parts)}.",
            f"    - 📊 **Resultado do filtro:** {total_results} evento(s) com data confirmada correspondem a este filtro.",
            f"    - ✨ **Destaques mostrados:** {shown_results} resultado(s) mais relevantes (janela {shown_from}-{shown_to}).",
        ]

    scope_parts = [normalized_filter if not date_filter else f"{normalized_filter} ({date_window})"]
    scope_parts.append(normalized_category if normalized_category else "all categories")
    if normalized_query:
        scope_parts.append(f"theme focus: {normalized_query}")
    else:
        scope_parts.append("broad event discovery")
    return [
        "- 🧾 **Search summary**",
        f"    - 🧭 **Filter used:** {', '.join(scope_parts)}.",
        f"    - 📊 **Result count:** {total_results} confirmed-date event(s) match this filter.",
        f"    - ✨ **Highlights shown:** {shown_results} most relevant result(s) (window {shown_from}-{shown_to}).",
    ]


_QUOTED_LOOKUP_PATTERN = re.compile(r'"([^"\n]{2,120})"|“([^”\n]{2,120})”')
_GENERIC_LOOKUP_MARKER_PATTERN = re.compile(
    r"\b(?:tell me about|tell me more about|more about|what about|how about|details(?: about| on)?|"
    r"information(?: about| on)?|info(?: about| on)?|about(?: the)?|sobre(?: o| a| os| as)?|"
    r"fala me de|diz me sobre|e do|e da|and what about|and how about)\b"
)


def _normalize_lookup_text(text: Optional[str]) -> str:
    """Normalizes lookup text for cross-language name matching."""
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _extract_lookup_tokens(text: Optional[str]) -> List[str]:
    """Extracts normalized alphanumeric tokens for title matching."""
    return re.findall(r"[a-z0-9]+", _normalize_lookup_text(text))


def _unique_lookup_tokens(text: Optional[str]) -> List[str]:
    """Returns unique normalized lookup tokens while preserving order."""
    return list(dict.fromkeys(_extract_lookup_tokens(text)))


def _minimum_token_similarity(query_token: str, candidate_token: str) -> float:
    """Returns the minimum similarity required for fuzzy token matches."""
    shortest_length = min(len(query_token), len(candidate_token))
    if shortest_length <= 3:
        return 1.0
    if shortest_length == 4:
        return 0.86
    if shortest_length <= 6:
        return 0.79
    return 0.74


def _token_similarity_score(query_token: str, candidate_token: str) -> float:
    """Scores token similarity, including prefix and typo-friendly matches."""
    if not query_token or not candidate_token:
        return 0.0
    if query_token == candidate_token:
        return 1.0
    if query_token.isdigit() or candidate_token.isdigit():
        return 0.0

    shorter_length = min(len(query_token), len(candidate_token))
    longer_length = max(len(query_token), len(candidate_token))
    length_ratio = shorter_length / longer_length if longer_length else 0.0

    if len(query_token) >= 4 and candidate_token.startswith(query_token) and length_ratio >= 0.6:
        return 0.97
    if len(candidate_token) >= 4 and query_token.startswith(candidate_token) and length_ratio >= 0.6:
        return 0.93
    if len(query_token) >= 5 and query_token in candidate_token and length_ratio >= 0.55:
        return 0.91
    if len(candidate_token) >= 5 and candidate_token in query_token and length_ratio >= 0.55:
        return 0.88

    similarity = SequenceMatcher(None, query_token, candidate_token).ratio()
    if similarity >= _minimum_token_similarity(query_token, candidate_token):
        return similarity
    return 0.0


def _best_token_similarity(query_token: str, candidate_tokens: List[str]) -> float:
    """Finds the best fuzzy score for a query token against candidate tokens."""
    best_score = 0.0
    for candidate_token in candidate_tokens:
        score = _token_similarity_score(query_token, candidate_token)
        if score > best_score:
            best_score = score
            if best_score >= 0.999:
                break
    return best_score


def _collect_token_match_stats(query_tokens: List[str], searchable_text: Optional[str]) -> Tuple[int, float]:
    """Returns matched-token count and weighted fuzzy score against candidate text."""
    if not query_tokens or not searchable_text:
        return 0, 0.0

    candidate_tokens = _unique_lookup_tokens(searchable_text)
    if not candidate_tokens:
        return 0, 0.0

    matched_count = 0
    weighted_score = 0.0
    for query_token in query_tokens:
        best_score = _best_token_similarity(query_token, candidate_tokens)
        if best_score > 0:
            matched_count += 1
            weighted_score += best_score

    return matched_count, weighted_score


def _phrase_similarity_score(query_phrase: Optional[str], candidate_phrase: Optional[str]) -> float:
    """Scores phrase similarity while being robust to spacing and punctuation differences."""
    normalized_query = _normalize_lookup_text(query_phrase)
    normalized_candidate = _normalize_lookup_text(candidate_phrase)
    if not normalized_query or not normalized_candidate:
        return 0.0
    if normalized_query == normalized_candidate:
        return 1.0
    if len(normalized_query) >= 5 and normalized_candidate.startswith(normalized_query):
        return 0.97
    if len(normalized_candidate) >= 5 and normalized_query.startswith(normalized_candidate):
        return 0.92

    compact_query = normalized_query.replace(" ", "")
    compact_candidate = normalized_candidate.replace(" ", "")
    if compact_query == compact_candidate:
        return 0.99
    if len(compact_query) >= 5 and compact_query in compact_candidate:
        return 0.9
    if len(compact_candidate) >= 5 and compact_candidate in compact_query:
        return 0.87

    similarity = SequenceMatcher(None, compact_query, compact_candidate).ratio()
    threshold = 0.78 if min(len(compact_query), len(compact_candidate)) <= 8 else 0.72
    return similarity if similarity >= threshold else 0.0


def _text_contains_fuzzy_term(text: Optional[str], terms: List[str] | set[str] | Tuple[str, ...]) -> bool:
    """Checks whether free text contains any term, tolerating minor typos and partial words."""
    normalized_text = _normalize_lookup_text(text)
    if not normalized_text:
        return False

    text_tokens = _unique_lookup_tokens(normalized_text)
    for raw_term in terms:
        normalized_term = _normalize_lookup_text(raw_term)
        if not normalized_term:
            continue
        if normalized_term in normalized_text:
            return True

        term_tokens = _unique_lookup_tokens(normalized_term)
        if not term_tokens:
            continue

        matched_count, weighted_score = _collect_token_match_stats(term_tokens, normalized_text)
        if matched_count == len(term_tokens) and weighted_score >= max(0.9, len(term_tokens) * 0.78):
            return True

        if len(term_tokens) == 1 and _best_token_similarity(term_tokens[0], text_tokens) > 0:
            return True

    return False


def _strip_lookup_year_tokens(text: Optional[str]) -> str:
    """Removes standalone year-like tokens from a lookup phrase."""
    normalized = _normalize_lookup_text(text)
    return re.sub(r"\b(?:\d{2}|(?:19|20)\d{2})\b", " ", normalized).strip()


def _extract_named_lookup_phrase(query: Optional[str], noise_tokens: set[str]) -> Optional[str]:
    """Extracts the specific named subject from quoted or 'tell me about' queries."""
    if not query:
        return None

    for raw_match in _QUOTED_LOOKUP_PATTERN.findall(query):
        candidate = next((part for part in raw_match if part), "")
        normalized_candidate = _normalize_lookup_text(candidate)
        if normalized_candidate:
            return normalized_candidate

    normalized_query = _normalize_lookup_text(query)
    if not _GENERIC_LOOKUP_MARKER_PATTERN.search(normalized_query):
        return None

    cleaned = _GENERIC_LOOKUP_MARKER_PATTERN.sub(" ", normalized_query)
    tokens = [token for token in _extract_lookup_tokens(cleaned) if token not in noise_tokens]
    if not tokens or len(tokens) > 8:
        return None

    return " ".join(tokens)


def _build_event_searchable_text(event: Dict[str, Any]) -> str:
    """Builds a richer event search blob including slug, venue, and links."""
    information_links = event.get("information_links")
    info_text = ""
    if isinstance(information_links, dict):
        info_text = " ".join(str(key) for key in information_links.keys())

    highlight_links = event.get("highlight_links")
    highlight_text = ""
    if isinstance(highlight_links, list):
        highlight_text = " ".join(
            filter(
                None,
                [
                    f"{item.get('title', '')} {item.get('url', '')}"
                    for item in highlight_links
                    if isinstance(item, dict)
                ],
            )
        )

    venue_locations = event.get("venue_locations")
    venue_text = ""
    if isinstance(venue_locations, list):
        venue_text = " ".join(
            filter(
                None,
                [
                    f"{item.get('venue_name', '')} {item.get('location', '')}"
                    for item in venue_locations
                    if isinstance(item, dict)
                ],
            )
        )

    schedule_notes = event.get("schedule_notes")
    schedule_text = ""
    if isinstance(schedule_notes, list):
        schedule_text = " ".join(str(note) for note in schedule_notes if note)

    title = event.get("title") or _clean_event_title(event.get("title"), event.get("url", ""))
    slug_title = _humanize_visitlisboa_slug(event.get("url", ""))
    return " ".join(
        filter(
            None,
            [
                title,
                slug_title,
                event.get("category", ""),
                event.get("location", ""),
                event.get("venue_name", ""),
                venue_text,
                event.get("short_description", ""),
                event.get("full_description", ""),
                schedule_text,
                info_text,
                highlight_text,
                event.get("url", ""),
            ],
        )
    )


def _flatten_text_values(value: Any) -> str:
    """Flattens nested dict/list values into a searchable text blob."""
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            parts.append(_flatten_text_values(key))
            parts.append(_flatten_text_values(val))
        return " ".join(part for part in parts if part)
    if isinstance(value, list):
        return " ".join(_flatten_text_values(item) for item in value)
    return str(value)


def _build_place_searchable_text(place: Dict[str, Any]) -> str:
    """Builds a richer place search blob including structured enriched metadata."""
    return " ".join(
        filter(
            None,
            [
                place.get('title', ''),
                place.get('category', ''),
                place.get('address', ''),
                place.get('location', ''),
                place.get('short_description', ''),
                place.get('full_description', ''),
                _flatten_text_values(place.get('features')),
                _flatten_text_values(place.get('contact_info')),
                _flatten_text_values(place.get('information_links')),
                _flatten_text_values(place.get('social_media')),
                _flatten_text_values(place.get('schedules')),
                _flatten_text_values(place.get('tickets_offers')),
                _flatten_text_values(place.get('additional_sections')),
                _flatten_text_values(place.get('tripadvisor')),
                place.get('lisboa_card_benefit', ''),
                place.get('lisboa_card_discount', ''),
                place.get('url', ''),
            ],
        )
    )


_EVENT_SPECIFIC_LOOKUP_NOISE_TOKENS = {
    "tell", "about", "details", "detail", "information", "info", "event", "events",
    "evento", "eventos", "more", "please", "show", "find", "me", "the", "this",
    "that", "these", "those", "what", "which", "and", "how", "sobre", "diz",
    "fala", "para", "from", "with", "there", "happening", "temos", "tem", "do",
    "da", "de", "dos", "das", "this", "week", "today", "tomorrow", "next", "year",
    "ano", "esta", "semana", "este", "proxima", "proximo",
}


def _extract_specific_event_lookup_phrase(query: Optional[str]) -> Optional[str]:
    """Extracts a specific event name from a natural-language query when present."""
    extracted = _extract_named_lookup_phrase(query, _EVENT_SPECIFIC_LOOKUP_NOISE_TOKENS)
    if extracted:
        return extracted

    raw_query = (query or '').strip()
    meaningful_tokens = [
        token for token in _extract_lookup_tokens(raw_query)
        if token not in _EVENT_SPECIFIC_LOOKUP_NOISE_TOKENS
    ]
    has_year_marker = bool(re.search(r"(?:'\d{2}\b|\b(?:19|20)\d{2}\b)", raw_query))
    has_title_like_casing = any(char.isupper() for char in raw_query)

    if meaningful_tokens and len(meaningful_tokens) <= 6 and (has_year_marker or has_title_like_casing):
        return " ".join(meaningful_tokens)

    return None


def _score_event_query_match(
    event: Dict[str, Any],
    expanded_tokens: List[str],
    specific_lookup_phrase: Optional[str] = None,
) -> float:
    """Scores how strongly an event matches the user's thematic or specific-name query."""
    if not expanded_tokens and not specific_lookup_phrase:
        return 0.0

    searchable = _normalize_lookup_text(_build_event_searchable_text(event))
    title_variants = {
        _normalize_lookup_text(event.get("title")),
        _normalize_lookup_text(_clean_event_title(event.get("title"), event.get("url", ""))),
        _normalize_lookup_text(_humanize_visitlisboa_slug(event.get("url", ""))),
    }
    title_variants.discard("")
    title_text = " ".join(sorted(title_variants))

    score = 0.0
    normalized_specific = _normalize_lookup_text(specific_lookup_phrase)
    normalized_specific_no_year = _strip_lookup_year_tokens(specific_lookup_phrase)

    if normalized_specific:
        if normalized_specific in title_variants:
            score += 140.0
        elif normalized_specific_no_year and normalized_specific_no_year in title_variants:
            score += 130.0
        elif any(normalized_specific in title for title in title_variants):
            score += 100.0
        elif normalized_specific_no_year and any(normalized_specific_no_year in title for title in title_variants):
            score += 92.0
        elif normalized_specific in searchable:
            score += 60.0
        elif normalized_specific_no_year and normalized_specific_no_year in searchable:
            score += 54.0

        phrase_candidate = normalized_specific_no_year or normalized_specific
        best_title_phrase_score = max(
            (
                max(
                    _phrase_similarity_score(phrase_candidate, title),
                    _phrase_similarity_score(phrase_candidate, _strip_lookup_year_tokens(title)),
                )
                for title in title_variants
            ),
            default=0.0,
        )
        if best_title_phrase_score > 0:
            score += 70.0 * best_title_phrase_score

        specific_tokens = [
            token for token in _extract_lookup_tokens(normalized_specific_no_year or normalized_specific)
            if not token.isdigit()
        ]
        if specific_tokens:
            title_token_hits, title_weighted_score = _collect_token_match_stats(specific_tokens, title_text)
            text_token_hits, text_weighted_score = _collect_token_match_stats(specific_tokens, searchable)
            if title_token_hits == len(specific_tokens):
                score += 48.0
            score += min(36.0, title_weighted_score * 10.0)
            score += min(18.0, text_weighted_score * 3.0)

    if expanded_tokens:
        title_hits, title_weighted_score = _collect_token_match_stats(expanded_tokens, title_text)
        text_hits, text_weighted_score = _collect_token_match_stats(expanded_tokens, searchable)
        score += min(32.0, title_weighted_score * 8.0)
        score += min(24.0, text_weighted_score * 3.0)

    return score


_EVENT_GENERIC_QUERY_TERMS = {
    'event', 'events', 'evento', 'eventos',
    'cultura', 'cultural', 'culture', 'culturais',
    'lisbon', 'lisboa', 'portugal', 'city', 'cidade',
    'great', 'major', 'grandes', 'explorar', 'explore',
    'find', 'finding', 'search', 'show', 'mostrar', 'mostra', 'encontra', 'encontre',
    'procura', 'procure', 'descobre', 'discover', 'want', 'quero',
    'this', 'week', 'esta', 'semana', 'what', 'which', 'que', 'quais',
    'there', 'happening', 'temos', 'have', 'local', 'locais',
    'tell', 'about', 'details', 'detail', 'information', 'info', 'specific',
    'sobre', 'diz', 'fala', 'me', 'more', 'please', 'year', 'ano',
}

_EVENT_QUERY_SYNONYMS = {
    'music': ['concert', 'concerto', 'live', 'band', 'artist', 'musical', 'fado', 'jazz', 'rock', 'pop'],
    'musica': ['music', 'concert', 'concerto', 'live', 'band', 'artist', 'musical', 'fado', 'jazz', 'rock', 'pop'],
    'música': ['music', 'concert', 'concerto', 'live', 'band', 'artist', 'musical', 'fado', 'jazz', 'rock', 'pop'],
    'concert': ['music', 'live', 'performance', 'show', 'gig'],
    'concerts': ['music', 'live', 'performance', 'show', 'gig'],
    'live': ['music', 'concert', 'performance', 'gig', 'show'],
    'vivo': ['live', 'music', 'concert', 'performance', 'gig', 'show'],
    'art': ['exhibition', 'gallery', 'museum', 'painting', 'sculpture', 'artwork'],
    'exhibition': ['art', 'gallery', 'museum', 'display', 'expo'],
    'theater': ['theatre', 'play', 'drama', 'stage', 'performance'],
    'theatre': ['theater', 'play', 'drama', 'stage', 'performance'],
    'dance': ['ballet', 'dancing', 'choreography', 'performance'],
    'family': ['children', 'kids', 'child', 'families'],
    'children': ['family', 'kids', 'child', 'families'],
    'kids': ['children', 'family', 'child', 'families'],
    'crianças': ['children', 'kids', 'family', 'families'],
    'criancas': ['children', 'kids', 'family', 'families'],
    'miúdos': ['children', 'kids', 'family', 'families'],
    'miudos': ['children', 'kids', 'family', 'families'],
    'night': ['nightlife', 'evening', 'live', 'late'],
    'nightlife': ['night', 'evening', 'live', 'bar'],
    'noite': ['night', 'nightlife', 'evening', 'live'],
    'food': ['gastronomy', 'culinary', 'wine', 'taste', 'restaurant'],
    'book': ['livro', 'literature', 'literary', 'reading'],
    'livro': ['book', 'literature', 'literary', 'reading'],
    'fair': ['feira', 'market', 'salon'],
    'feira': ['fair', 'market', 'salon'],
}


def _expand_event_query_tokens(query: Optional[str]) -> List[str]:
    """Builds a small expanded token set for event text matching."""
    if not query:
        return []

    normalized_query = _normalize_lookup_text(query)
    normalized_query = re.sub(r"\bmusica\s+ao\s+vivo\b", "live music", normalized_query)
    normalized_query = re.sub(r"\bao\s+vivo\b", "live", normalized_query)
    original_tokens = [token for token in re.findall(r"[a-z0-9]+", normalized_query) if len(token) >= 3]
    query_tokens = [t for t in original_tokens if t not in _EVENT_GENERIC_QUERY_TERMS]

    if not query_tokens:
        return []

    expanded_tokens = set(query_tokens)
    for token in query_tokens:
        expanded_tokens.update(_EVENT_QUERY_SYNONYMS.get(token, []))

    return sorted(expanded_tokens)


def _event_matches_query(event: Dict[str, Any], expanded_tokens: List[str]) -> bool:
    """Returns whether an event matches the expanded query tokens."""
    if not expanded_tokens:
        return True

    return _score_event_query_match(event, expanded_tokens) > 0


# ==========================================================================
# Vector Store Connection (Lazy Loading)
# ==========================================================================

_vector_store = None


def _get_vector_store():
    """
    Lazily initializes the vector store connection.

    Returns:
        KnowledgeBase: The vector store instance, or None if unavailable.
    """
    global _vector_store

    if _vector_store is None:
        try:
            # Import here to avoid circular imports and slow startup
            from tools.vector_store import KnowledgeBase

            # Initialize with CPU to avoid GPU memory issues in agent context
            _vector_store = KnowledgeBase(use_gpu=False)
            logger.info("✅ Vector store initialized for semantic search")
        except Exception as e:
            logger.warning(f"⚠️ Vector store unavailable: {e}")
            _vector_store = False  # Mark as unavailable

    return _vector_store if _vector_store else None


# Expose initialization for external use (e.g., app startup)
initialize_vector_store = _get_vector_store


# ==========================================================================
# Hybrid Search: VisitLisboa + Dados Abertos
# ==========================================================================

# Keywords that trigger Dados Abertos search (in addition to VisitLisboa)
DADOS_ABERTOS_KEYWORDS = {
    # Shopping & Commerce
    'shopping', 'centro comercial', 'mall', 'mercado', 'feira', 'quiosque', 'loja',
    # Health & Emergency
    'hospital', 'urgência', 'urgencias', 'saude', 'saúde', 'clinica', 'clínica',
    'farmacia', 'farmácia', 'pharmacy', 'pharmacies', 'bombeiros', 'firefighters', 'policia', 'polícia', 'police', 'segurança', 'seguranca',
    # Education
    'escola', 'school', 'schools', 'colegio', 'colégio', 'universidade', 'university', 'faculdade', 'faculty', 'instituto', 'institute', 'creche',
    # Culture (complement)
    'biblioteca', 'library', 'libraries', 'teatro', 'cinema', 'galeria', 'gallery', 'monumento', 'miradouro', 'igreja',
    # Outdoors & Leisure
    'jardim', 'garden', 'gardens', 'parque', 'park', 'parks', 'piscina', 'pool', 'desporto',
    # Services & Amenities
    'wc', 'banheiro', 'sanitário', 'sanitario', 'estacionamento', 'parking',
    'embaixada', 'cemiterio', 'cemitério', 'junta', 'câmara', 'camara',
    # Streets & Locations
    'rua', 'avenida', 'praça', 'praca', 'largo', 'bairro'
}


def _matches_open_data_keyword(query: Optional[str], keyword: str) -> bool:
    """Checks open-data trigger keywords with conservative typo tolerance."""
    normalized_query = _normalize_lookup_text(query)
    normalized_keyword = _normalize_lookup_text(keyword)
    if not normalized_query or not normalized_keyword:
        return False
    if normalized_keyword in normalized_query:
        return True

    query_tokens = _unique_lookup_tokens(normalized_query)
    keyword_tokens = _unique_lookup_tokens(normalized_keyword)
    if not keyword_tokens:
        return False

    if len(keyword_tokens) > 1:
        matched_count, weighted_score = _collect_token_match_stats(keyword_tokens, normalized_query)
        return matched_count == len(keyword_tokens) and weighted_score >= len(keyword_tokens) * 0.92

    keyword_token = keyword_tokens[0]
    if len(keyword_token) <= 4:
        return keyword_token in query_tokens

    return _best_token_similarity(keyword_token, query_tokens) >= 0.84


def _should_search_dados_abertos(query: Optional[str]) -> bool:
    """
    Checks if query contains keywords that warrant searching Dados Abertos.

    Args:
        query: The search query.

    Returns:
        bool: True if Dados Abertos should be searched.
    """
    if not query:
        return False

    return any(_matches_open_data_keyword(query, keyword) for keyword in DADOS_ABERTOS_KEYWORDS)


def _score_open_data_place_match(query: str, name: str, address: str = "") -> float:
    """Scores open-data matches so exact named facilities outrank generic early matches."""
    normalized_query = _normalize_place_hint_text(query)
    name_text = _normalize_place_hint_text(name)
    address_text = _normalize_place_hint_text(address)
    combined_text = f"{name_text} {address_text}".strip()

    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", normalized_query)
        if len(token) >= 3 and token not in _GENERIC_PLACE_QUERY_TOKENS
    ]

    score = 0.0
    if normalized_query and normalized_query in name_text:
        score += 12.0
    elif normalized_query and normalized_query in combined_text:
        score += 8.0

    phrase_score = _phrase_similarity_score(normalized_query, name_text)
    if phrase_score > 0:
        score += 9.0 * phrase_score

    for token in tokens:
        token_name_score = _best_token_similarity(token, _unique_lookup_tokens(name_text))
        token_address_score = _best_token_similarity(token, _unique_lookup_tokens(address_text))
        token_combined_score = _best_token_similarity(token, _unique_lookup_tokens(combined_text))
        if token_name_score > 0:
            score += 3.0 * token_name_score
        elif token_address_score > 0:
            score += 1.5 * token_address_score
        elif token_combined_score > 0:
            score += 1.0 * token_combined_score

    return score


def _search_dados_abertos_hybrid(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches Dados Abertos and returns structured results for merging.

    Args:
        query: Search query.
        max_results: Maximum results.

    Returns:
        List of place dictionaries compatible with VisitLisboa format.
    """
    try:
        from tools.dados_abertos import (
            DF_METADATA,
            extract_address,
            extract_coordinates,
            extract_name,
            fetch_geojson_with_retry,
            search_datasets,
        )

        if DF_METADATA.empty:
            return []

        query_lower = query.lower()
        normalized_query = _normalize_place_hint_text(query)
        query_tokens = [t for t in re.findall(r"[a-z0-9]+", normalized_query) if len(t) >= 3]
        found_places = []

        # Keyword mapping (simplified from dados_abertos.py)
        keyword_map = {
            'hospital': ['Hospitais Públicos', 'Hospitais Privados', 'Centros de Saúde'],
            'farmacia': ['Farmácias e Parafarmácias'],
            'escola': ['Escolas Públicas - 1º Ciclo', 'Escolas Públicas - 2º e 3º Ciclo', 'Escolas Públicas - Secundário'],
            'universidade': ['Ensino Superior', 'Faculdades, Escolas e Institutos'],
            'faculdade': ['Ensino Superior', 'Faculdades, Escolas e Institutos'],
            'biblioteca': ['Bibliotecas Arquivos e Centros de Documentação'],
            'jardim': ['Jardins - Parques Urbanos', 'Grandes Parques e Jardins de Lisboa'],
            'parque': ['Grandes Parques e Jardins de Lisboa', 'Parques Infantis'],
            'mercado': ['Mercados', 'Feiras'],
            'centro comercial': ['Centros Comerciais'],
            'shopping': ['Centros Comerciais'],
            'policia': ['Polícia Municipal', 'Polícia de Segurança Pública'],
            'bombeiros': ['Bombeiros'],
            'miradouro': ['Miradouros'],
            'estacionamento': ['Parques de estacionamento na via pública'],
            'wc': ['Instalações Sanitárias'],
            'piscina': ['Instalações Desportivas'],
        }

        # Find matching datasets
        potential_datasets = []
        for key, titles in keyword_map.items():
            if key in query_lower:
                for title in titles:
                    matches = DF_METADATA[DF_METADATA['title'] == title]
                    for _, row in matches.iterrows():
                        if row.get('stable_url') and row.get('stable_url') != "N/A":
                            potential_datasets.append(row)

        # Also do keyword search
        tokens = [w for w in query_lower.split() if len(w) > 3]
        for token in tokens:
            matches = search_datasets(token)
            for _, row in matches.head(3).iterrows():
                if row.get('stable_url') and row.get('stable_url') != "N/A":
                    potential_datasets.append(row)

        # Deduplicate
        seen_urls = set()
        unique_datasets = []
        for ds in potential_datasets:
            url = ds.get('stable_url')
            if url not in seen_urls:
                seen_urls.add(url)
                unique_datasets.append(ds)

        # Fetch and search (limit to 3 datasets for speed)
        for dataset in unique_datasets[:3]:
            title = dataset.get('title', 'Unknown')
            url = dataset.get('stable_url')

            if not url:
                continue

            # Skip large/irrelevant datasets
            if any(x in title.lower() for x in ['limites', 'rede viária', 'carta', 'zonamento']):
                continue

            data = fetch_geojson_with_retry(url)
            if not data:
                continue

            features = data.get('features', [])
            for feature in features[:200]:  # Limit per dataset while allowing better ranking
                properties = feature.get('properties', {})
                name = extract_name(properties)

                if name == "N/A":
                    continue

                # Check match
                address = extract_address(properties)
                match_score = _score_open_data_place_match(query, name, address)

                if match_score > 0 or any(t in _normalize_place_hint_text(name) for t in query_tokens):
                    coords = extract_coordinates(feature.get('geometry', {}))

                    found_places.append({
                        'title': name,
                        'category': f"📊 Open Data: {title}",
                        'location': address if address != "N/A" else "Lisboa",
                        'short_description': f"From Lisboa Aberta dataset: {title}",
                        'url': None,
                        'lat': coords[0] if coords else None,
                        'lon': coords[1] if coords else None,
                        'source': 'dados_abertos',
                        '_match_score': match_score,
                    })

        # Rank and deduplicate by title, keeping the strongest match
        found_places.sort(key=lambda item: item.get('_match_score', 0), reverse=True)
        unique = {}
        for p in found_places:
            existing = unique.get(p['title'])
            if existing is None or p.get('_match_score', 0) > existing.get('_match_score', 0):
                unique[p['title']] = p

        ranked_results = list(unique.values())[:max_results]
        for item in ranked_results:
            item.pop('_match_score', None)

        return ranked_results

    except Exception as e:
        logger.warning(f"Dados Abertos hybrid search failed: {e}")
        return []


# ==========================================================================
# Fallback: Direct JSON Loading
# ==========================================================================

def _load_events_json() -> List[Dict[str, Any]]:
    """Loads events directly from JSON as fallback."""
    try:
        with open(Config.PATH_VISIT_LISBOA_EVENTS, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def _load_places_json() -> List[Dict[str, Any]]:
    """Loads places directly from JSON as fallback."""
    try:
        with open(Config.PATH_VISIT_LISBOA_PLACES, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


# Cached places data for enrichment lookup
_places_cache: Optional[Dict[str, Dict]] = None


def _get_place_by_url(url: str) -> Optional[Dict[str, Any]]:
    """
    Looks up full place data from JSON by URL.
    Uses caching to avoid repeated file reads.

    Args:
        url: The VisitLisboa URL of the place.

    Returns:
        Full place dictionary or None if not found.
    """
    global _places_cache

    if _places_cache is None:
        places = _load_places_json()
        _places_cache = {p.get('url', ''): p for p in places if p.get('url')}

    return _places_cache.get(url)


# ==========================================================================
# Helper Functions
# ==========================================================================

def _extract_event_from_doc(doc) -> str:
    """
    Formats a vector store document as an event summary.

    Args:
        doc: LangChain Document from vector store.

    Returns:
        str: Formatted event summary.
    """
    parts = []
    metadata = doc.metadata
    content = doc.page_content

    # Title
    title = metadata.get('title', 'Unknown Event')
    parts.append(f"📅 **{title}**")

    # Category
    category = metadata.get('category', 'General')
    if category and category != 'General':
        parts.append(f"   Category: {category}")

    # Extract key info from content
    content_lines = content.split('\n')
    for line in content_lines[:5]:  # First 5 lines
        if line.startswith('Name:'):
            continue  # Skip, already have title
        if len(line) > 200:
            line = line[:200] + "..."
        if line.strip():
            parts.append(f"   {line}")

    # URL
    url = metadata.get('url', '')
    if url:
        parts.append(f"   🔗 {url}")

    return "\n".join(parts)


def _extract_place_from_doc(doc) -> str:
    """
    Formats a vector store document as a place summary.

    Args:
        doc: LangChain Document from vector store.

    Returns:
        str: Formatted place summary.
    """
    parts = []
    metadata = doc.metadata
    content = doc.page_content

    # Title
    title = metadata.get('title', 'Unknown Place')
    parts.append(f"🏛️ **{title}**")

    # Category
    category = metadata.get('category', 'General')
    if category and category != 'General':
        parts.append(f"   Category: {category}")

    # Extract key info from content
    content_lines = content.split('\n')
    for line in content_lines[:5]:  # First 5 lines
        if line.startswith('Name:'):
            continue  # Skip, already have title
        if len(line) > 200:
            line = line[:200] + "..."
        if line.strip():
            parts.append(f"   {line}")

    # URL
    url = metadata.get('url', '')
    if url:
        parts.append(f"   🔗 {url}")

    return "\n".join(parts)


_PLACE_CATEGORY_ALIASES = {
    "museums & monuments": {
        "museums & monuments", "museum", "museums", "monument", "monuments",
        "museu", "museus", "monumento", "monumentos", "monastery", "castle",
        "palace", "church",
    },
    "restaurants": {"restaurant", "restaurants", "restaurante", "restaurantes", "food", "dining", "gastronomy", "gastronomia"},
    "hotels": {"hotel", "hotels", "guest house", "guest houses", "apartments", "accommodation", "alojamento"},
    "view points": {"view point", "view points", "viewpoint", "viewpoints", "miradouro", "miradouros"},
    "parks & gardens": {"park", "parks", "garden", "gardens", "parque", "parques", "jardim", "jardins", "nature"},
    "tours": {"tour", "tours", "trip", "trips"},
}

_GENERIC_PLACE_QUERY_TOKENS = {
    "best", "good", "top", "lisbon", "lisboa", "place", "places", "attraction",
    "attractions", "nearby", "today", "see", "visit", "thing", "things", "related",
    "wheelchair", "accessible", "accessibility", "stepfree", "step", "mobility",
    "what", "where", "which", "when", "are", "the", "and", "for", "with",
    "from", "that", "this", "these", "those", "into", "about", "around",
    "tell", "details", "detail", "information", "info", "more", "please",
    "museum", "museums", "museu", "museus", "monument", "monuments",
    "monumento", "monumentos", "open", "opened", "closed", "like",
    "imperdiveis", "imperdíveis", "primeira", "first", "time", "trip",
    "visitor", "visitors", "visita", "visiting", "must", "mustsee",
}
_KNOWN_PLACE_LOCATION_HINTS = {
    "belem", "alfama", "chiado", "baixa", "rossio", "oriente", "expo",
    "ajuda", "alcantara", "estrela", "graca", "mouraria", "restelo",
    "beato", "cascais", "sintra", "campo", "sodre",
}
_EXPLICIT_MUSEUM_MARKERS = {
    "museum", "museu", "maat", "mude", "gulbenkian", "berardo", "mac/ccb", "macccb",
}
_NON_MUSEUM_MONUMENT_MARKERS = {
    "monument", "monastery", "castle", "palace", "church", "tower", "cemetery", "aqueduct",
    "monumento", "mosteiro", "castelo", "palacio", "igreja", "torre", "cemiterio", "aqueduto",
}
_PUBLIC_SERVICE_FOCUS_TERMS = {
    "hospital": {"hospital", "hospitals", "urgencia", "urgencias", "urgência", "urgências"},
    "pharmacy": {"pharmacy", "pharmacies", "farmacia", "farmacias", "farmácia", "farmácias", "parafarmacia", "parafarmácia"},
    "school": {"school", "schools", "escola", "escolas", "college", "colegio", "colégio"},
    "university": {"university", "universities", "universidade", "universidades", "faculty", "faculdade", "instituto", "institute"},
    "library": {"library", "libraries", "biblioteca", "bibliotecas"},
    "police": {"police", "policia", "polícia"},
    "firefighters": {"fire", "firefighters", "firefighter", "bombeiro", "bombeiros"},
    "parking": {"parking", "estacionamento", "parque de estacionamento", "car park"},
    "market": {"market", "markets", "mercado", "mercados", "feira", "feiras"},
    "garden": {"garden", "gardens", "jardim", "jardins", "park", "parks", "parque", "parques"},
    "wc": {"wc", "toilet", "toilets", "sanitario", "sanitários", "sanitario", "sanitário", "restroom"},
    "embassy": {"embassy", "embassies", "embaixada", "embaixadas"},
}
_OUTSIDE_LISBON_CITY_MARKERS = {
    "cascais", "sintra", "almada", "setubal", "setúbal", "oeiras", "amadora",
    "loures", "odivelas", "montijo", "seixal", "sesimbra", "barreiro", "mafra",
    "alcochete", "moita", "palmela", "vila franca", "vila franca de xira",
    "santa iria", "azóia", "azoia",
}


def _normalize_place_hint_text(text: Optional[str]) -> str:
    """Normalizes text for location-hint comparisons."""
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    return normalized.lower()


def _extract_place_location_hints(query: Optional[str]) -> List[str]:
    """Extracts known Lisbon area hints from the user query."""
    normalized_query = _normalize_place_hint_text(query)
    tokens = re.findall(r"[a-z0-9]+", normalized_query)
    return [token for token in tokens if token in _KNOWN_PLACE_LOCATION_HINTS]


def _matches_place_location_hints(text: str, location_hints: List[str]) -> bool:
    """Checks whether candidate text matches requested neighborhood/location hints."""
    if not location_hints:
        return True

    normalized_text = _normalize_place_hint_text(text)
    return any(hint in normalized_text for hint in location_hints)


def _query_requests_ranked_places(query: Optional[str]) -> bool:
    """Detects broad ranking intents such as 'best museums' or 'top places'."""
    return _text_contains_fuzzy_term(query, ["best", "top", "recommended", "recommend", "must-see", "must see"])


def _is_explicit_museum_candidate(title: str, url: str = "", extra_text: str = "") -> bool:
    """Returns whether a candidate is explicitly museum-like rather than just monument-like."""
    extra_text_clean = re.sub(r"category\s*:\s*[^\n]+", " ", extra_text or "", flags=re.IGNORECASE)
    extra_text_clean = re.sub(r"museums?\s*&\s*monuments?", " ", extra_text_clean, flags=re.IGNORECASE)
    haystack = _normalize_place_hint_text(" ".join([title or "", url or "", extra_text_clean]))
    compact_haystack = haystack.replace(" ", "")

    if any(marker in haystack for marker in _EXPLICIT_MUSEUM_MARKERS):
        return True
    if any(marker in compact_haystack for marker in {"mac/ccb", "macccb"}):
        return True
    if any(marker in haystack for marker in _NON_MUSEUM_MONUMENT_MARKERS):
        return False
    return False


def _normalize_place_category_filter(category: Optional[str]) -> Optional[str]:
    """Normalizes user-facing category filters to a canonical key."""
    if not category:
        return None

    normalized = str(category).strip().lower()
    for canonical, aliases in _PLACE_CATEGORY_ALIASES.items():
        if normalized == canonical or normalized in aliases:
            return canonical
    return normalized


def _place_category_matches(item_category: str, requested_category: Optional[str]) -> bool:
    """Returns whether a place category matches the requested user filter."""
    if not requested_category:
        return True

    normalized_requested = _normalize_place_category_filter(requested_category)
    item_category_lower = (item_category or "").lower()
    if not normalized_requested:
        return True

    aliases = _PLACE_CATEGORY_ALIASES.get(normalized_requested, {normalized_requested})
    return any(alias in item_category_lower for alias in aliases)


def _extract_place_query_tokens(query: Optional[str]) -> List[str]:
    """Extracts meaningful query tokens for post-retrieval reranking."""
    if not query:
        return []

    tokens = _extract_lookup_tokens(query)
    return [
        token for token in tokens
        if len(token) >= 3 and token not in _GENERIC_PLACE_QUERY_TOKENS
    ]


def _extract_specific_place_lookup_phrase(query: Optional[str]) -> Optional[str]:
    """Extracts a specific place name from quoted or 'tell me about' queries."""
    extracted = _extract_named_lookup_phrase(query, _GENERIC_PLACE_QUERY_TOKENS)
    if extracted:
        return extracted

    raw_query = (query or '').strip()
    meaningful_tokens = [
        token for token in _extract_lookup_tokens(raw_query)
        if token not in _GENERIC_PLACE_QUERY_TOKENS
    ]
    has_title_like_casing = any(char.isupper() for char in raw_query)
    title_like_tokens = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'/-]*", raw_query)
    capitalized_token_count = sum(1 for token in title_like_tokens if token and token[0].isupper())
    requires_title_majority = math.ceil(len(title_like_tokens) / 2) if title_like_tokens else 0

    if (
        meaningful_tokens
        and len(meaningful_tokens) <= 5
        and has_title_like_casing
        and title_like_tokens
        and capitalized_token_count >= max(1, requires_title_majority)
    ):
        return " ".join(meaningful_tokens)
    return None


def _clean_place_description_text(text: Optional[str], fallback_title: str = "") -> str:
    """Removes raw metadata scaffolding from place descriptions."""
    if not text:
        return ""

    cleaned_parts: List[str] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lower_line = line.lower()
        if any(lower_line.startswith(prefix) for prefix in ["name:", "url:", "category:", "address:", "location:"]):
            continue
        if lower_line.startswith("short description:"):
            line = line.split(":", 1)[1].strip()
        if fallback_title and _normalize_place_hint_text(line) == _normalize_place_hint_text(fallback_title):
            continue
        cleaned_parts.append(line)

    cleaned = re.sub(r"\s+", " ", " ".join(cleaned_parts)).strip(" -")
    return cleaned


_PLACE_QUERY_DAY_ALIASES = {
    "monday": ["monday", "segunda"],
    "tuesday": ["tuesday", "terça", "terca"],
    "wednesday": ["wednesday", "quarta"],
    "thursday": ["thursday", "quinta"],
    "friday": ["friday", "sexta"],
    "saturday": ["saturday", "sábado", "sabado"],
    "sunday": ["sunday", "domingo"],
}


def _query_mentions_lisboa_card(query: Optional[str]) -> bool:
    """Returns whether the query explicitly asks about Lisboa Card benefits."""
    return _text_contains_fuzzy_term(query, ["lisboa card", "free with lisboa card", "discount", "with lisboa card"])


def _query_mentions_tickets(query: Optional[str]) -> bool:
    """Returns whether the query explicitly asks about tickets or offers."""
    return _text_contains_fuzzy_term(query, ["ticket", "tickets", "bilhete", "bilhetes", "offer", "offers", "price", "prices"])


def _query_mentions_schedule(query: Optional[str]) -> bool:
    """Returns whether the query explicitly asks about schedules or opening hours."""
    return _text_contains_fuzzy_term(
        query,
        [
            "schedule", "schedules", "opening hours", "hours", "open", "opens", "closed",
            "horário", "horários", "horarios", "abre", "aberto", "aberta", "fechado", "fechada",
            "today", "tomorrow", "hoje", "amanhã", "amanha",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "segunda", "terça", "terca", "quarta", "quinta", "sexta", "sábado", "sabado", "domingo",
        ],
    )


def _extract_requested_schedule_days(query: Optional[str]) -> List[str]:
    """Extracts requested weekdays from a schedule-related place query."""
    normalized_query = _normalize_place_hint_text(query)
    requested_days = []
    for canonical_day, aliases in _PLACE_QUERY_DAY_ALIASES.items():
        if any(alias in normalized_query for alias in aliases):
            requested_days.append(canonical_day)
    return requested_days


def _schedule_day_matches(day_label: str, canonical_day: str) -> bool:
    """Checks whether a scraped day label matches a canonical weekday."""
    normalized_label = _normalize_place_hint_text(day_label)
    return any(alias in normalized_label for alias in _PLACE_QUERY_DAY_ALIASES.get(canonical_day, []))


def _format_place_schedule_previews(
    schedules: List[Dict[str, Any]],
    query: Optional[str],
) -> List[str]:
    """Builds concise schedule preview lines relevant to the user's query."""
    if not schedules:
        return []

    requested_days = _extract_requested_schedule_days(query)
    previews: List[str] = []

    if requested_days:
        for schedule in schedules:
            schedule_name = schedule.get('name')
            for requested_day in requested_days:
                for day_label, hours in (schedule.get('hours') or {}).items():
                    if _schedule_day_matches(day_label, requested_day):
                        if schedule_name and str(schedule_name).lower() != 'schedule':
                            previews.append(f"{schedule_name}: {day_label} {hours}")
                        else:
                            previews.append(f"{day_label}: {hours}")
        if previews:
            return previews[:2]

    for schedule in schedules:
        if schedule.get('today'):
            previews.append(schedule['today'])
            break

    if previews:
        return previews[:1]

    # Fallback: show first two hours entries when no today shortcut is available.
    for schedule in schedules:
        for day_label, hours in list((schedule.get('hours') or {}).items())[:2]:
            previews.append(f"{day_label}: {hours}")
        if previews:
            break

    if previews:
        return previews[:2]

    for schedule in schedules:
        if schedule.get('summary'):
            previews.append(schedule['summary'])
            break

    return previews[:1]


def _extract_required_service_term_groups(query: Optional[str]) -> List[set[str]]:
    """Extracts critical service-intent term groups that must remain present in results."""
    required_groups: List[set[str]] = []
    for variants in _PUBLIC_SERVICE_FOCUS_TERMS.values():
        if any(_matches_open_data_keyword(query, term) for term in variants):
            required_groups.append(variants)
    return required_groups


def _matches_required_service_terms(searchable_text: str, required_groups: List[set[str]]) -> bool:
    """Checks whether candidate text preserves the critical service intent from the query."""
    if not required_groups:
        return True

    return any(
        _text_contains_fuzzy_term(searchable_text, variants)
        for variants in required_groups
    )


def _query_explicitly_mentions_outside_lisbon(query: Optional[str]) -> bool:
    """Returns whether the user explicitly asked for an area outside Lisbon city."""
    normalized_query = _normalize_place_hint_text(query)
    return any(marker in normalized_query for marker in _OUTSIDE_LISBON_CITY_MARKERS)


def _place_within_requested_geography(location_text: str, query: Optional[str]) -> bool:
    """Keeps Lisbon-city queries focused on Lisbon unless the user explicitly asked otherwise."""
    if not location_text:
        return True
    if _query_explicitly_mentions_outside_lisbon(query):
        return True

    normalized_location = _normalize_place_hint_text(location_text)
    return not any(marker in normalized_location for marker in _OUTSIDE_LISBON_CITY_MARKERS)


def _normalize_place_result_key(place: Dict[str, Any]) -> str:
    """Builds a robust deduplication key for place results."""
    url = place.get("url") or ""
    if url:
        return _normalize_place_hint_text(url.rsplit("/", 1)[-1])

    title = _normalize_place_hint_text(place.get("title", ""))
    location = _normalize_place_hint_text(place.get("location", ""))
    return f"{title}|{location}"


def _append_unique_place_results(
    target_results: List[Dict[str, Any]],
    new_results: List[Dict[str, Any]],
    seen_keys: set[str],
    limit: Optional[int] = None,
) -> None:
    """Appends place results while deduplicating by URL/title-location signature."""
    for result in new_results:
        key = _normalize_place_result_key(result)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        target_results.append(result)
        if limit is not None and len(target_results) >= limit:
            break


def _convert_raw_place_to_result(place: Dict[str, Any], source: str = "visitlisboa") -> Dict[str, Any]:
    """Converts a raw VisitLisboa place record into the common result shape."""
    location = place.get("address") or place.get("location") or "Lisbon"
    description = _clean_place_description_text(
        place.get("short_description") or place.get("full_description") or "",
        place.get("title", ""),
    )

    return {
        "title": place.get("title", "Unknown"),
        "category": place.get("category", "General"),
        "location": location,
        "short_description": description,
        "url": place.get("url", ""),
        "source": source,
    }


def _is_service_like_place_category(item_category: str) -> bool:
    """Detects accommodation/info-desk categories that should not dominate museum queries."""
    category_lower = (item_category or "").lower()
    service_like_terms = [
        "hotel", "tourist office", "guest house", "apartments",
        "accommodation", "local & rural accommodation",
    ]
    return any(term in category_lower for term in service_like_terms)


def _is_non_attraction_category(item_category: str) -> bool:
    """Detects categories that should not dominate broad first-time attraction lists."""
    category_lower = (item_category or "").lower()
    excluded_terms = [
        "tour", "tours", "hotel", "guest house", "accommodation",
        "tourist office", "apartments", "nightlife",
    ]
    return any(term in category_lower for term in excluded_terms)


def _top_attraction_category_bonus(item_category: str) -> float:
    """Scores iconic sightseeing categories higher for broad attraction queries."""
    category_lower = (item_category or "").lower()
    if any(term in category_lower for term in ["museum", "monument", "monastery", "castle", "palace", "church"]):
        return 0.35
    if any(term in category_lower for term in ["view point", "viewpoint", "miradouro"]):
        return 0.24
    if any(term in category_lower for term in ["park", "garden", "jardim", "parque"]):
        return 0.10
    if any(term in category_lower for term in ["family", "kids"]):
        return 0.04
    return 0.0


def _infer_place_query_intent(query: Optional[str], category: Optional[str]) -> Optional[str]:
    """Infers broad place-search intent for lightweight category exclusions."""
    normalized_category = _normalize_place_category_filter(category)
    museum_terms = ["museum", "museu", "museums", "museus"]
    monument_terms = ["monument", "monumento", "monuments", "monumentos", "monastery", "castle", "palace", "church"]
    top_attraction_terms = [
        "atrações imperdíveis", "atracoes imperdiveis", "must-see", "must see",
        "first time", "primeira vez", "top attractions", "main attractions",
        "highly recommended attractions", "o que visitar",
    ]

    museum_requested = _text_contains_fuzzy_term(query, museum_terms)
    monument_requested = _text_contains_fuzzy_term(query, monument_terms)
    top_attractions_requested = _text_contains_fuzzy_term(query, top_attraction_terms)

    if normalized_category == "museums & monuments" and museum_requested and not monument_requested:
        return "museum_only"
    if normalized_category == "museums & monuments":
        return "museum_monument"
    if museum_requested and not monument_requested:
        return "museum_only"
    if monument_requested and not museum_requested:
        return "monument_only"
    if museum_requested or monument_requested:
        return "museum_monument"
    if top_attractions_requested:
        return "top_attractions"
    if _text_contains_fuzzy_term(query, ["restaurant", "restaurante", "food", "dinner", "lunch", "brunch", "gastronomy"]):
        return "food"
    if _text_contains_fuzzy_term(query, ["hotel", "stay", "accommodation", "guest house"]):
        return "accommodation"
    return None


def _infer_event_date_filter_from_query(query: Optional[str]) -> Optional[str]:
    """Infers lightweight date filters directly from common PT/EN event phrasing."""
    normalized_query = _normalize_lookup_text(query)
    if not normalized_query:
        return None

    mappings = [
        (["this weekend", "este fim de semana", "fim de semana", "weekend"], "this weekend"),
        (["next week", "proxima semana", "próxima semana"], "next week"),
        (["this week", "esta semana"], "this week"),
        (["tomorrow", "amanha", "amanhã"], "tomorrow"),
        (["today", "hoje"], "today"),
        (["next month", "proximo mes", "próximo mês"], "next month"),
        (["this month", "este mes", "este mês"], "this month"),
        (["next year", "proximo ano", "próximo ano"], "next year"),
        (["this year", "este ano"], "this year"),
    ]

    for terms, inferred_filter in mappings:
        if any(term in normalized_query for term in terms):
            return inferred_filter
    return None


def _extract_knowledge_doc_snippet(doc: Any, max_chars: int = 220) -> str:
    """Builds a concise snippet from a vector-store document for user-facing summaries."""
    content = str(getattr(doc, "page_content", "") or "")
    if not content:
        return ""

    prioritized_prefixes = (
        "short description:",
        "full description:",
        "details:",
        "description:",
        "location:",
    )
    ignored_prefixes = (
        "name:",
        "url:",
        "image urls:",
        "video urls:",
        "indexed at:",
        "content hash:",
        "source:",
    )

    candidate_lines: List[str] = []
    fallback_lines: List[str] = []
    for raw_line in content.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        lower_line = line.lower()
        if lower_line.startswith(ignored_prefixes):
            continue
        if lower_line.startswith(prioritized_prefixes):
            candidate_lines.append(line.split(":", 1)[1].strip())
            continue

        fallback_lines.append(line)

    snippet = " ".join(candidate_lines[:2]) if candidate_lines else " ".join(fallback_lines[:2])
    snippet = re.sub(r"\s+", " ", snippet).strip(" -")
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    return snippet


def _infer_lisbon_knowledge_focus(query: Optional[str]) -> str:
    """Infers which collection should be emphasized in mixed knowledge search."""
    if _text_contains_fuzzy_term(
        query,
        [
            "lisboa card", "airport", "aeroporto", "metro", "tram", "bus", "train",
            "public transport", "transport", "how to get", "getting from", "go from",
            "city center", "city centre", "centro da cidade", "carris", "cp", "fertagus",
        ],
    ):
        return "guide"
    if _text_contains_fuzzy_term(
        query,
        [
            "event", "events", "concert", "festival", "exhibition", "show", "feira",
            "evento", "eventos", "teatro", "music", "música", "musica",
        ],
    ):
        return "events"
    if _text_contains_fuzzy_term(
        query,
        [
            "museum", "museu", "restaurant", "restaurante", "hotel", "viewpoint",
            "miradouro", "place", "places", "attraction", "where to eat", "eat",
        ],
    ):
        return "places"
    return "general"


def _matches_place_query_intent(item_category: str, searchable_text: str, query_intent: Optional[str]) -> bool:
    """Applies lightweight intent filtering to remove obvious false positives."""
    if not query_intent:
        return True

    haystack = f"{item_category or ''} {searchable_text or ''}".lower()
    museum_markers = ["museum", "museu", "museums"]
    monument_markers = ["monument", "monumento", "monastery", "castle", "palace", "church"]

    if query_intent == "museum_only":
        return any(marker in haystack for marker in museum_markers)
    if query_intent == "monument_only":
        return any(marker in haystack for marker in monument_markers)
    if query_intent == "museum_monument":
        return any(marker in haystack for marker in museum_markers + monument_markers)
    if query_intent == "top_attractions":
        return not _is_non_attraction_category(item_category)

    return True


def _fallback_search(query: str, category: str, data: List[Dict], max_results: int) -> List[Dict]:
    """
    Fallback text search when vector store is unavailable.

    Args:
        query: Search query.
        category: Category filter.
        data: JSON data list.
        max_results: Maximum results.

    Returns:
        List of matching items.
    """
    results = []
    query_lower = query.lower() if query else None
    query_tokens = _extract_place_query_tokens(query)
    query_intent = _infer_place_query_intent(query, category)
    location_hints = _extract_place_location_hints(query)
    required_service_terms = _extract_required_service_term_groups(query)
    normalized_query = _normalize_lookup_text(query)
    specific_lookup_phrase = _extract_specific_place_lookup_phrase(query)

    for item in data:
        # Category filter
        if category and not _place_category_matches(item.get('category', ''), category):
            continue

        searchable = _build_place_searchable_text(item).lower()
        normalized_searchable = _normalize_place_hint_text(searchable)
        service_anchor_text = " ".join([
            item.get('title', ''),
            item.get('category', ''),
            item.get('location', ''),
            item.get('short_description', ''),
        ])

        if not _place_within_requested_geography(item.get('location', ''), query):
            continue

        if not _matches_required_service_terms(service_anchor_text, required_service_terms):
            continue

        if not _matches_place_location_hints(searchable, location_hints):
            continue

        if query_intent == "museum_only" and not _is_explicit_museum_candidate(
            item.get('title', ''),
            item.get('url', ''),
            searchable,
        ):
            continue

        if not _matches_place_query_intent(item.get('category', ''), item.get('title', ''), query_intent):
            continue

        # Query filter
        if query_lower:
            if query_tokens:
                matched_tokens, weighted_score = _collect_token_match_stats(query_tokens, normalized_searchable)
                phrase_score = _phrase_similarity_score(specific_lookup_phrase or normalized_query, item.get('title', ''))
                if matched_tokens == 0 and weighted_score <= 0 and phrase_score <= 0:
                    continue
            elif query_lower not in searchable:
                phrase_score = _phrase_similarity_score(normalized_query, item.get('title', ''))
                if phrase_score <= 0:
                    continue

        results.append(item)
        if len(results) >= max_results:
            break

    return results


# ==========================================================================
# LangChain Tools
# ==========================================================================

@tool
def search_cultural_events(
    query: Optional[str] = None,
    category: Optional[str] = None,
    date_filter: Optional[str] = None,
    max_results: int = 10,
    offset: int = 0,
    language: Optional[str] = None,
) -> str:
    """
    Search for cultural events in Lisbon with DATE FILTERING.

    This tool finds events and ALWAYS filters by date. If no date is specified,
    defaults to upcoming events in the next 30 days.

    Args:
        query (str, optional): Natural language query describing what you're looking for.
            Examples: 'music concerts', 'art exhibitions', 'family activities',
                     'outdoor events', 'Christmas celebrations', 'fado music'.
        category (str, optional): Filter by event category. Options include:
            'Main Events', 'Exhibitions', 'Music', 'Theater', 'Dance',
            'Cinema', 'Sports', 'Fairs', 'Festivals', 'Gastronomy'.
        date_filter (str, optional): Date range filter. CRITICAL for temporal queries.
            Options: 'today', 'tomorrow', 'this week', 'next week', 'this weekend',
                    'this month', 'next month', 'January', 'February', etc.
                    Also accepts: 'hoje', 'amanhã', 'esta semana', 'próxima semana',
                    'este fim de semana', 'este mês', 'próximo mês'.
                    Or ISO date: '2025-01-15'.
            Default: 'upcoming' (next 30 days).
        max_results (int): Maximum number of results to return (default: 10).
        offset (int): Number of matching results to skip before returning the next batch.

    Returns:
        str: Formatted list of matching events with dates and links.

    Examples:
        - search_cultural_events(query="concerts", date_filter="next week")
        - search_cultural_events(category="Exhibitions", date_filter="this month")
        - search_cultural_events(date_filter="this weekend") -> Weekend events
    """
    try:
        # Normalize inputs
        query = str(query).strip() if query and str(query).strip() and str(query).lower() != 'none' else None
        category = str(category).strip() if category and str(category).strip() and str(category).lower() != 'none' else None
        date_filter = str(date_filter).strip() if date_filter and str(date_filter).strip() and str(date_filter).lower() != 'none' else None

        if not isinstance(max_results, int) or max_results <= 0:
            max_results = 10
        if not isinstance(offset, int) or offset < 0:
            offset = 0

        render_language = _infer_visitlisboa_output_language(query, language)
        specific_lookup_phrase = _extract_specific_event_lookup_phrase(query)
        effective_query = specific_lookup_phrase or query

        # Parse date range. Broad discovery defaults to upcoming, but specific lookups should search across all dates.
        if not date_filter and not specific_lookup_phrase:
            date_filter = _infer_event_date_filter_from_query(query) or 'upcoming'

        start_date, end_date = parse_date_range(date_filter) if date_filter else (None, None)

        # Logging
        date_info = f"{start_date.strftime('%Y-%m-%d') if start_date else 'any'} to {end_date.strftime('%Y-%m-%d') if end_date else 'any'}"
        logger.info(
            f"search_cultural_events: query='{query}', category='{category}', effective_query='{effective_query}', dates={date_info}, max={max_results}, offset={offset}"
        )

        # ALWAYS load JSON for date filtering (vector store doesn't filter by date)
        all_events_data = _load_events_json()
        events_data = list(all_events_data)

        if not events_data:
            return "❌ Events data not available."

        undated_candidates = [event for event in all_events_data if not get_event_dates(event)]

        # Step 1: Filter by date FIRST (most important)
        if start_date or end_date:
            events_data = filter_events_by_date(events_data, start_date, end_date)
            logger.info(f"After date filter: {len(events_data)} events")

        if not events_data:
            # No events in date range
            today = datetime.now()
            if render_language == "pt":
                return (
                    f"❌ Não encontrei eventos com data confirmada para o filtro '{date_filter}'.\n"
                    f"🧭 **Filtro aplicado:** {date_filter} ({date_info}).\n"
                    f"📅 **Data de referência:** {today.strftime('%Y-%m-%d')}\n\n"
                    "💡 Experimente uma janela temporal mais alargada, como 'este mês' ou 'próximo mês'."
                )
            return (
                f"❌ No confirmed-date events found for the '{date_filter}' filter.\n"
                f"🧭 **Filter used:** {date_filter} ({date_info}).\n"
                f"📅 **Reference date:** {today.strftime('%Y-%m-%d')}\n\n"
                "💡 Try a broader date window such as 'this month' or 'next month'."
            )

        # Step 2: Filter by category
        if category:
            category_lower = category.lower()
            events_data = [e for e in events_data if category_lower in e.get('category', '').lower()]
            undated_candidates = [e for e in undated_candidates if category_lower in e.get('category', '').lower()]
            logger.info(f"After category filter: {len(events_data)} events")

        # Step 3: Filter by query (TOKEN-BASED matching for better recall)
        query_scores: Dict[int, float] = {}
        if query:
            expanded_tokens = _expand_event_query_tokens(effective_query)

            if specific_lookup_phrase or expanded_tokens:
                def _score_collection(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[int, float]]:
                    scores: Dict[int, float] = {}
                    matched_items: List[Dict[str, Any]] = []
                    for item in items:
                        score = _score_event_query_match(
                            item,
                            expanded_tokens,
                            specific_lookup_phrase=specific_lookup_phrase,
                        )
                        if score > 0:
                            scores[id(item)] = score
                            matched_items.append(item)
                    return matched_items, scores

                events_data, query_scores = _score_collection(events_data)
                undated_candidates, _ = _score_collection(undated_candidates)
                logger.info(
                    f"After query filter: {len(events_data)} events (specific_lookup={bool(specific_lookup_phrase)}, tokens={expanded_tokens[:6]})"
                )
            else:
                logger.info(f"Query '{query}' contained only generic terms, skipping text filter.")

        if not events_data:
            localized_date_filter = _localize_event_date_filter(date_filter, language=render_language)
            localized_category = _localize_event_category(category, language=render_language) if category else None
            if render_language == "pt":
                message = (
                    "❌ Não encontrei eventos com data confirmada que correspondam ao filtro pedido.\n\n"
                    f"🧭 **Filtro aplicado:** {localized_date_filter} ({date_info}), {localized_category or 'todas as categorias'}"
                )
                if query:
                    message += f", foco temático: {query}"
                message += ".\n\n💡 Experimente termos mais abrangentes, como 'música', 'arte' ou 'festival'."
            else:
                message = (
                    "❌ No confirmed-date events matched the requested filter.\n\n"
                    f"🧭 **Filter used:** {localized_date_filter} ({date_info}), {localized_category or 'all categories'}"
                )
                if query:
                    message += f", theme focus: {query}"
                message += ".\n\n💡 Try broader terms such as 'music', 'art', or 'festival'."
            if undated_candidates:
                if render_language == "pt":
                    message += (
                        "\n\n⚠️ **Nota sobre a completude da fonte:** "
                        f"{len(undated_candidates)} registo(s) adicional(is) compatíveis foram excluídos porque a fonte ainda não confirma a respetiva data."
                    )
                else:
                    message += (
                        "\n\n⚠️ **Source completeness note:** "
                        f"{len(undated_candidates)} additional matching record(s) were excluded because the source does not confirm their dates yet."
                    )
            return message

        # Step 4: SORT BY TEMPORAL RELEVANCE (CRITICAL!)
        # Ephemeral events (single-day concerts) should rank ABOVE long exhibitions
        for event in events_data:
            event['_relevance_score'] = calculate_temporal_relevance_score(event, start_date, end_date)
            event['_duration_days'] = get_event_duration_days(event)
            event['_query_match_score'] = query_scores.get(id(event), 0.0)

        if query_scores:
            events_data.sort(
                key=lambda e: (e.get('_query_match_score', 0.0), e.get('_relevance_score', 0.0)),
                reverse=True,
            )
            logger.info(
                "Sorted by query relevance first (top query score: %.1f, top temporal score: %.1f)",
                events_data[0].get('_query_match_score', 0.0),
                events_data[0].get('_relevance_score', 0.0),
            )
        else:
            events_data.sort(key=lambda e: e.get('_relevance_score', 0), reverse=True)
            logger.info(f"Sorted by temporal relevance (top score: {events_data[0].get('_relevance_score', 0):.1f})")

        if offset >= len(events_data):
            output_parts = _format_event_filter_summary(
                query=query,
                category=category,
                date_filter=date_filter,
                start_date=start_date,
                end_date=end_date,
                total_results=len(events_data),
                shown_results=0,
                offset=offset,
                language=render_language,
            )
            if render_language == "pt":
                output_parts.extend(
                    [
                        "",
                        "❌ Já não há mais eventos para mostrar com este filtro.",
                    ]
                )
            else:
                output_parts.extend(
                    [
                        "",
                        "❌ There are no more events to show for this filter window.",
                    ]
                )
            return "\n".join(output_parts).strip()

        # Limit results
        results = events_data[offset : offset + max_results]

        # Format output with contextual filter summary and concise descriptions
        output_parts = _format_event_filter_summary(
            query=query,
            category=category,
            date_filter=date_filter,
            start_date=start_date,
            end_date=end_date,
            total_results=len(events_data),
            shown_results=len(results),
            offset=offset,
            language=render_language,
        )
        output_parts.append("")

        for i, event in enumerate(results, 1):
            title = _clean_event_title(event.get('title'), event.get('url', ''))
            cat = _localize_event_category(event.get('category', 'General'), language=render_language)
            loc = event.get('location', 'Lisbon')
            venue_name = str(event.get('venue_name') or '').strip()
            if venue_name and venue_name.lower() not in loc.lower():
                loc = f"{venue_name}, {loc}"
            dates_str = format_event_dates(event, language=render_language)
            duration = event.get('_duration_days', get_event_duration_days(event))
            duration_label = _format_event_duration_label(duration, language=render_language)
            description_summary = _summarize_event_description(event.get('full_description'))
            price_text = _localize_event_price(event.get('price'), language=render_language)

            output_parts.append(f"{i}. 📅 **{title}**")
            if render_language == "pt":
                output_parts.append(f"   🗓️ **Quando:** {dates_str}")
                output_parts.append(f"   ⏱️ **Duração:** {duration_label}")
                output_parts.append(f"   📂 **Categoria:** {cat}")
            else:
                output_parts.append(f"   🗓️ **When:** {dates_str}")
                output_parts.append(f"   ⏱️ **Duration:** {duration_label}")
                output_parts.append(f"   📂 **Category:** {cat}")

            if description_summary:
                if render_language == "pt":
                    output_parts.append(f"   📝 **Descrição:** {description_summary}")
                else:
                    output_parts.append(f"   📝 **Description:** {description_summary}")

            output_parts.append(f"   📍 {loc}")

            # Show price information if available
            if price_text:
                if render_language == "pt":
                    output_parts.append(f"   💰 **Preço:** {price_text}")
                else:
                    output_parts.append(f"   💰 **Price:** {price_text}")

            if event.get('url'):
                output_parts.append(f"   🔗 {event['url']}")

            # Show buy tickets link if available
            if event.get('buy_tickets_url'):
                if render_language == "pt":
                    output_parts.append(f"   🎟️ **Comprar bilhetes:** {event['buy_tickets_url']}")
                else:
                    output_parts.append(f"   🎟️ **Buy tickets:** {event['buy_tickets_url']}")

            if event.get('schedule_notes'):
                schedule_summary = "; ".join(str(note) for note in event['schedule_notes'][:2])
                if render_language == "pt":
                    output_parts.append(f"   🕐 **Horários:** {schedule_summary}")
                else:
                    output_parts.append(f"   🕐 **Schedule:** {schedule_summary}")

            if event.get('highlight_links'):
                highlight_titles = ", ".join(
                    str(item.get('title', ''))
                    for item in event['highlight_links'][:3]
                    if isinstance(item, dict) and item.get('title')
                )
                if highlight_titles:
                    if render_language == "pt":
                        output_parts.append(f"   ✨ **Destaques:** {highlight_titles}")
                    else:
                        output_parts.append(f"   ✨ **Highlights:** {highlight_titles}")

            output_parts.append("")  # Empty line between events

        if len(events_data) > max_results:
            if render_language == "pt":
                output_parts.append(
                    f"💡 A lista mostra {len(results)} destaque(s). Se quiser, posso refinar a pesquisa com uma categoria, bairro ou data mais específica."
                )
            else:
                output_parts.append(
                    f"💡 This list shows {len(results)} highlights. Narrow the search by category, area, or date for a tighter selection."
                )
        if undated_candidates:
            if render_language == "pt":
                output_parts.append(
                    "⚠️ **Nota sobre a completude da fonte:** "
                    f"{len(undated_candidates)} registo(s) adicional(is) compatíveis foram excluídos porque a fonte ainda não confirma a respetiva data."
                )
            else:
                output_parts.append(
                    "⚠️ **Source completeness note:** "
                    f"{len(undated_candidates)} additional matching record(s) were excluded because the source does not confirm their dates yet."
                )

        return "\n".join(output_parts)

    except Exception as e:
        logger.error(f"Error in search_cultural_events: {e}")
        return f"❌ Error searching events: {str(e)}"


@tool
def search_places_attractions(
    query: Optional[str] = None,
    category: Optional[str] = None,
    max_results: int = 10,
    offset: int = 0,
    language: Optional[str] = None,
) -> str:
    """
    Search for places and attractions in Lisbon using HYBRID search.

    This tool combines:
    1. VisitLisboa semantic search (tourist attractions, restaurants, hotels)
    2. Dados Abertos (public infrastructure: hospitals, schools, parks, etc.)

    Args:
        query (str, optional): Natural language query describing what you're looking for.
            Examples: 'museums with art', 'good restaurants for dinner',
                     'places to see sunset', 'historic monuments', 'beaches nearby',
                     'hospital urgências', 'escola secundária', 'jardim parque'.
        category (str, optional): Filter by place category. Options include:
            'Museums & Monuments', 'Restaurants', 'Hotels', 'View Points',
            'Beaches', 'Shopping', 'Nightlife', 'Parks & Gardens', 'Tours'.
        max_results (int): Maximum number of results to return (default: 10).
        offset (int): Number of matching results to skip before returning the next batch.
        language (str, optional): Preferred output language (`pt` or `en`).

    Returns:
        str: Formatted list of matching places with descriptions and links.

    Examples:
        - search_places_attractions(query="tower") -> Belém Tower, etc.
        - search_places_attractions(category="Museums") -> All museums
        - search_places_attractions(query="hospital santa maria") -> Hybrid results
    """
    try:
        # Normalize inputs
        query = str(query).strip() if query and str(query).strip() and str(query).lower() != 'none' else None
        category = str(category).strip() if category and str(category).strip() and str(category).lower() != 'none' else None

        if not isinstance(max_results, int) or max_results <= 0:
            max_results = 10
        if not isinstance(offset, int) or offset < 0:
            offset = 0

        requested_window = max_results + offset
        render_language = _infer_visitlisboa_output_language(query, language)
        specific_lookup_query = _extract_specific_place_lookup_phrase(query)
        effective_query = specific_lookup_query or query
        query_intent = _infer_place_query_intent(effective_query or query, category)

        logger.info(
            f"search_places_attractions: query='{query}', effective_query='{effective_query}', category='{category}', max={max_results}, offset={offset}"
        )
        required_service_terms = _extract_required_service_term_groups(effective_query or query)

        # Check if we should also search Dados Abertos (hybrid mode)
        search_dados_abertos = _should_search_dados_abertos(effective_query or query)
        dados_abertos_results = []

        if search_dados_abertos and (effective_query or query):
            logger.info("Hybrid mode: Query contains Dados Abertos keywords")
            dados_abertos_results = _search_dados_abertos_hybrid(effective_query or query, max_results=requested_window // 2 + 1)
            logger.info(f"Dados Abertos returned {len(dados_abertos_results)} results")

        # =====================================================================
        # STEP 1: Search VisitLisboa (Vector Store)
        # =====================================================================
        visitlisboa_results = []
        kb = _get_vector_store()

        if kb:
            try:
                requested_category = _normalize_place_category_filter(category)
                search_query = effective_query or query or "places and attractions in Lisbon"
                if query_intent == "top_attractions":
                    search_query = f"iconic attractions monuments viewpoints historic sites {search_query}"
                if category:
                    category_prefix = category
                    if requested_category == "museums & monuments" and query_intent == "museum_only":
                        category_prefix = "Museums"
                    search_query = f"{category_prefix} {search_query}"

                query_tokens = _extract_place_query_tokens(effective_query or search_query)
                location_hints = _extract_place_location_hints(effective_query or search_query)
                ranking_requested = _query_requests_ranked_places(effective_query or search_query) or query_intent == "top_attractions"
                lisboa_card_requested = _query_mentions_lisboa_card(effective_query or query)
                tickets_requested = _query_mentions_tickets(effective_query or query)
                schedule_requested = _query_mentions_schedule(effective_query or query)

                logger.info(f"search_places_attractions: searching VisitLisboa for '{search_query}'")

                results_with_scores = kb.search_with_scores(
                    query=search_query,
                    k=requested_window * 2,
                    collections=[COLLECTION_PLACES]
                )

                RELEVANCE_THRESHOLD = 1.8
                # Fetch more results to allow for re-ranking
                candidate_pool = [r for r in results_with_scores if r[1] <= RELEVANCE_THRESHOLD]

                logger.info(f"VisitLisboa: {len(results_with_scores)} raw, {len(candidate_pool)} candidates for ranking")

                scored_candidates = []

                for doc, vector_score in candidate_pool:
                    metadata = doc.metadata
                    item_category = metadata.get('category', 'General')

                    if requested_category and not _place_category_matches(item_category, requested_category):
                        continue

                    if query_intent == "museum_monument" and _is_service_like_place_category(item_category):
                        continue

                    cleaned_doc_description = _clean_place_description_text(doc.page_content, metadata.get('title', ''))
                    searchable = f"{metadata.get('title', '')} {item_category} {doc.page_content}".lower()
                    normalized_title = _normalize_place_hint_text(metadata.get('title', ''))
                    normalized_searchable = _normalize_place_hint_text(searchable)
                    raw_location = metadata.get('address') or metadata.get('location') or ''
                    full_place_data = _get_place_by_url(metadata.get('url', '')) if metadata.get('url') else None
                    location_text = f"{metadata.get('title', '')} {metadata.get('url', '')} {raw_location} {doc.page_content}".lower()
                    service_anchor_text = " ".join([
                        metadata.get('title', ''),
                        item_category,
                        metadata.get('url', ''),
                        raw_location,
                        cleaned_doc_description,
                    ])

                    if not _place_within_requested_geography(raw_location, query):
                        continue

                    if not _matches_required_service_terms(service_anchor_text, required_service_terms):
                        continue

                    if not _matches_place_location_hints(location_text, location_hints):
                        continue

                    if query_intent == "museum_only" and not _is_explicit_museum_candidate(
                        metadata.get('title', ''),
                        metadata.get('url', ''),
                        searchable,
                    ):
                        continue

                    if not _matches_place_query_intent(item_category, metadata.get('title', ''), query_intent):
                        continue

                    token_hits, token_weighted_score = _collect_token_match_stats(query_tokens, normalized_searchable)
                    title_hits, title_weighted_score = _collect_token_match_stats(query_tokens, normalized_title)
                    phrase_score = _phrase_similarity_score(effective_query or search_query, metadata.get('title', ''))

                    if query_tokens and token_hits == 0 and phrase_score <= 0 and vector_score > 1.25:
                        continue

                    # Calculate Rank Score
                    # Formula: (relevance * 0.6) + (rating * 0.3) + (log(reviews) * 0.1)

                    # 1. Relevance: vector_score is distance (lower=better). Invert it.
                    # typical distance 0.2 to 1.5.
                    relevance_val = 1.0 / (1.0 + vector_score)

                    # 2. Rating: Try to find distinct rating in metadata (or default 3.0)
                    # Note: Metadata usually lacks deep rating info unless enriched.
                    # We'll default to 0 if missing, to penalty unrated places slightly, or 3.0 neutral
                    rating_val = float(metadata.get('rating', 0)) or 3.0
                    rating_norm = rating_val / 5.0

                    # 3. Reviews: Log10
                    reviews_val = int(metadata.get('reviews', 0))
                    reviews_log = math.log10(reviews_val + 1) / 5.0  # Max typical ~5

                    if ranking_requested:
                        relevance_weight, rating_weight, reviews_weight = 0.25, 0.45, 0.30
                    else:
                        relevance_weight, rating_weight, reviews_weight = 0.60, 0.30, 0.10

                    category_bonus = 0.22 if requested_category and _place_category_matches(item_category, requested_category) else 0.0
                    title_bonus = min(0.18, title_weighted_score * 0.09)
                    token_bonus = min(0.15, token_weighted_score * 0.05)
                    service_penalty = 0.25 if query_tokens and _is_service_like_place_category(item_category) and query_intent != "accommodation" else 0.0
                    museum_specific_bonus = 0.10 if query_intent == "museum_only" and _is_explicit_museum_candidate(
                        metadata.get('title', ''),
                        metadata.get('url', ''),
                        searchable,
                    ) else 0.0
                    lisboa_card_bonus = 0.18 if lisboa_card_requested and full_place_data and (
                        full_place_data.get('lisboa_card_benefit') or full_place_data.get('lisboa_card_discount')
                    ) else 0.0
                    tickets_bonus = 0.12 if tickets_requested and full_place_data and (
                        full_place_data.get('tickets_offers') or full_place_data.get('contact_info', {}).get('tickets_url')
                    ) else 0.0
                    schedule_bonus = 0.12 if schedule_requested and full_place_data and full_place_data.get('schedules') else 0.0
                    top_attraction_bonus = _top_attraction_category_bonus(item_category) if query_intent == "top_attractions" else 0.0
                    phrase_bonus = min(0.16, phrase_score * 0.16)
                    iconic_title_bonus = 0.0
                    if query_intent == "top_attractions":
                        iconic_markers = [
                            "belem", "jeronimos", "castelo", "sao jorge", "santa justa",
                            "oceanario", "miradouro", "gulbenkian", "maat", "alfama",
                        ]
                        if any(marker in normalized_title for marker in iconic_markers):
                            iconic_title_bonus = 0.08

                    final_score = (
                        (relevance_val * relevance_weight)
                        + (rating_norm * rating_weight)
                        + (reviews_log * reviews_weight)
                        + category_bonus
                        + title_bonus
                        + token_bonus
                        + museum_specific_bonus
                        + lisboa_card_bonus
                        + tickets_bonus
                        + schedule_bonus
                        + top_attraction_bonus
                        + phrase_bonus
                        + iconic_title_bonus
                        - service_penalty
                    )

                    scored_candidates.append({
                        'doc': doc,
                        'vector_score': vector_score,
                        'final_score': final_score,
                        'metadata': metadata,
                        'cleaned_doc_description': cleaned_doc_description,
                    })

                # Sort by FINAL SCORE descending
                scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)

                # Convert to standard format
                for item in scored_candidates[:requested_window]:
                    metadata = item['metadata']
                    # Attempt to get real address/location
                    real_location = metadata.get('address') or metadata.get('location') or 'Lisbon'
                    visitlisboa_results.append({
                        'title': metadata.get('title', 'Unknown'),
                        'category': metadata.get('category', 'General'),
                        'location': real_location,
                        'short_description': item['cleaned_doc_description'],
                        'url': metadata.get('url', ''),
                        'source': 'visitlisboa',
                        'score': item['vector_score'],  # Keep original for debug if needed
                        'ranking_score': item['final_score']
                    })

            except Exception as e:
                logger.warning(f"Vector search failed: {e}")

        # JSON fallback to improve recall when vector search under-recovers
        should_use_json_fallback = bool(effective_query or query) and (
            not visitlisboa_results
            or bool(specific_lookup_query)
            or query_intent != "top_attractions"
        )
        if should_use_json_fallback and len(visitlisboa_results) < requested_window:
            fallback_items = _fallback_search(
                query=effective_query or query,
                category=category,
                data=_load_places_json(),
                max_results=requested_window * 2,
            )
            fallback_results = [_convert_raw_place_to_result(item) for item in fallback_items]
            combined_visitlisboa: List[Dict[str, Any]] = []
            seen_visitlisboa_keys: set[str] = set()
            _append_unique_place_results(combined_visitlisboa, visitlisboa_results, seen_visitlisboa_keys)
            _append_unique_place_results(combined_visitlisboa, fallback_results, seen_visitlisboa_keys, limit=requested_window)
            visitlisboa_results = combined_visitlisboa

        if required_service_terms:
            visitlisboa_results = [
                result
                for result in visitlisboa_results
                if _matches_required_service_terms(
                    " ".join(
                        [
                            result.get('title', ''),
                            result.get('category', ''),
                            result.get('url', ''),
                            result.get('location', ''),
                            result.get('short_description', ''),
                        ]
                    ),
                    required_service_terms,
                )
            ]

        # =====================================================================
        # STEP 2: Merge Results (VisitLisboa + Dados Abertos)
        # =====================================================================

        # HYBRID STRATEGY: Interleave results from both sources
        # For queries matching Dados Abertos keywords, prioritize those results
        # since the user is likely looking for public infrastructure
        all_results = []
        existing_keys: set[str] = set()

        if search_dados_abertos and dados_abertos_results:
            # User searched for infrastructure -> prioritize Dados Abertos
            # Take half from Dados Abertos, half from VisitLisboa
            da_quota = requested_window // 2 + 1
            vl_quota = requested_window - da_quota + 1

            # Add Dados Abertos first (more relevant for infrastructure queries)
            _append_unique_place_results(all_results, dados_abertos_results[:da_quota], existing_keys)

            # Then add VisitLisboa (tourist-focused, but may have relevant results)
            _append_unique_place_results(all_results, visitlisboa_results[:vl_quota], existing_keys)
        else:
            # Standard tourist query -> prioritize VisitLisboa
            _append_unique_place_results(all_results, visitlisboa_results, existing_keys)

            # Add any Dados Abertos results that don't duplicate
            _append_unique_place_results(all_results, dados_abertos_results, existing_keys)

        # =====================================================================
        # STEP 3: Format Output
        # =====================================================================

        if not all_results:
            # Last resort fallback
            if effective_query or query:
                fallback_items = _fallback_search(effective_query or query, category, _load_places_json(), max_results=requested_window)
                if fallback_items:
                    all_results = [_convert_raw_place_to_result(item) for item in fallback_items]

            if not all_results and (effective_query or query):
                logger.info("No results from hybrid search, trying direct Dados Abertos")
                from tools.dados_abertos import _search_place_in_datasets_logic
                open_data_results = _search_place_in_datasets_logic(effective_query or query, max_results=requested_window)
                if open_data_results:
                    return open_data_results

            if render_language == "pt":
                return f"Não foram encontrados locais correspondentes a '{query or 'todos'}' no VisitLisboa ou nos registos de dados abertos."
            return f"No places found matching: '{query or 'all'}' in VisitLisboa or Open Data registries."

        # Limit to max_results
        final_results = all_results[offset : offset + max_results]
        if not final_results:
            if render_language == "pt":
                return (
                    f"🧭 **Janela de resultados:** {offset + 1}-{offset + max_results} de {len(all_results)}.\n\n"
                    "❌ Já não há mais locais para mostrar com este filtro."
                )
            return (
                f"🧭 **Result window:** {offset + 1}-{offset + max_results} of {len(all_results)}.\n\n"
                "❌ There are no more places to show for this filter window."
            )

        # Count sources
        vl_count = sum(1 for r in final_results if r.get('source') == 'visitlisboa')
        da_count = sum(1 for r in final_results if r.get('source') == 'dados_abertos')

        if render_language == "pt":
            output_parts = [
                f"🏛️ **Found {len(final_results)} Places/Attractions in Lisbon:**\n",
                f"🧭 **Janela de resultados:** {offset + 1}-{offset + len(final_results)} de {len(all_results)}.",
            ]
        else:
            output_parts = [
                f"🏛️ **Found {len(final_results)} Places/Attractions in Lisbon:**\n",
                f"🧭 **Result window:** {offset + 1}-{offset + len(final_results)} of {len(all_results)}.",
            ]

        for i, place in enumerate(final_results, 1):
            title = place.get('title', 'Unknown')
            cat = place.get('category', 'General')
            loc = place.get('location', 'Lisbon')
            source = place.get('source', 'unknown')

            # Try to get full data from JSON for richer output
            full_data = None
            if place.get('url') and source == 'visitlisboa':
                full_data = _get_place_by_url(place['url'])

            # Source indicator
            if source == 'dados_abertos':
                output_parts.append(f"\n{i}. 📊 **{title}**")  # Open Data icon
            else:
                output_parts.append(f"\n{i}. 🏛️ **{title}**")  # VisitLisboa icon

            output_parts.append(f"   📂 Category: {cat}")

            # Lisboa Card benefit (from enriched data)
            lisboa_card_benefit = None
            if full_data:
                lisboa_card_benefit = full_data.get('lisboa_card_benefit') or full_data.get('lisboa_card_discount')
            if lisboa_card_benefit:
                output_parts.append(f"   🎫 {lisboa_card_benefit}")

            description_text = _clean_place_description_text(
                (full_data.get('short_description') if full_data else None) or place.get('short_description'),
                title,
            )
            if description_text:
                desc = description_text[:200]
                if len(description_text) > 200:
                    desc += "..."
                output_parts.append(f"   {desc}")

            output_parts.append(f"   📍 {loc}")

            # Schedule/opening hours (from enriched data)
            if full_data and full_data.get('schedules'):
                for preview in _format_place_schedule_previews(full_data['schedules'], effective_query or query):
                    output_parts.append(f"   🕐 {preview}")

            # Tickets/prices (from enriched data)
            if full_data and full_data.get('tickets_offers'):
                tickets = full_data['tickets_offers']
                if tickets.get('description'):
                    price_desc = tickets['description'][:80]
                    if len(tickets['description']) > 80:
                        price_desc += "..."
                    output_parts.append(f"   💰 {price_desc}")
                elif tickets.get('links'):
                    first_link = tickets['links'][0]
                    output_parts.append(f"   🎟️ {first_link.get('text', 'Tickets')}: {first_link.get('url', '')}")
            elif full_data and full_data.get('contact_info', {}).get('tickets_url') and _query_mentions_tickets(effective_query or query):
                output_parts.append(f"   🎟️ Tickets: {full_data['contact_info']['tickets_url']}")

            # TripAdvisor rating (from enriched data)
            if full_data and full_data.get('tripadvisor'):
                ta = full_data['tripadvisor']
                if ta.get('rating'):
                    output_parts.append(f"   ⭐ TripAdvisor: {ta['rating']}/5 ({ta.get('reviews_count', '?')} reviews)")

            # Contact info (from enriched data)
            if full_data and full_data.get('contact_info'):
                contact = full_data['contact_info']
                if contact.get('phone'):
                    output_parts.append(f"   📞 {contact['phone']}")

            if place.get('url'):
                output_parts.append(f"   🔗 {place['url']}")

            if place.get('lat') and place.get('lon'):
                output_parts.append(f"   🗺️ GPS: ({place['lat']:.5f}, {place['lon']:.5f})")

        # Source breakdown
        output_parts.append(f"\n\n📊 **Sources:** {vl_count} from VisitLisboa, {da_count} from Lisboa Aberta")
        if search_dados_abertos:
            output_parts.append("🔄 **Hybrid search:** Public infrastructure included")
        output_parts.append("💡 Try more specific queries for better results.")

        return "\n".join(output_parts)

    except Exception as e:
        logger.error(f"Error in search_places_attractions: {e}")
        return f"❌ Error searching places: {str(e)}"


@tool
def get_event_categories() -> str:
    """
    Get all available event categories from VisitLisboa data.

    Use this tool to discover what types of events are available
    before searching for specific events.

    Returns:
        str: List of event categories with counts.
    """
    try:
        logger.info("get_event_categories called")

        events_data = _load_events_json()

        if not events_data:
            return "❌ Events data not available."

        # Count events by category
        category_counts = {}
        for event in events_data:
            cat = event.get('category', 'Uncategorized')
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Sort by count
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

        output_parts = ["🎭 **Event Categories in Lisbon:**\n"]
        for cat, count in sorted_categories:
            output_parts.append(f"  • {cat}: {count} events")

        output_parts.append(f"\n📊 **Total events:** {len(events_data)}")
        output_parts.append("\n💡 Podes perguntar-me sobre um tipo de evento específico para pesquisa detalhada.")

        return "\n".join(output_parts)

    except Exception as e:
        logger.error(f"Error in get_event_categories: {e}")
        return f"❌ Error getting event categories: {str(e)}"


@tool
def get_place_categories() -> str:
    """
    Get all available place categories from VisitLisboa data.

    Use this tool to discover what types of places/attractions are available
    before searching for specific places.

    Returns:
        str: List of place categories with counts.
    """
    try:
        logger.info("get_place_categories called")

        places_data = _load_places_json()

        if not places_data:
            return "❌ Places data not available."

        category_counts = {}
        output_parts = ["🏛️ **Available Place Categories:**\n"]

        # Count places by category
        for place in places_data:
            cat = place.get('category', 'Uncategorized')
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Sort by count
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

        for cat, count in sorted_categories[:20]:  # Top 20
            output_parts.append(f"  • {cat}: {count} places")

        if len(sorted_categories) > 20:
            output_parts.append(f"  ... and {len(sorted_categories) - 20} more categories")

        output_parts.append(f"\n📊 **Total places:** {len(places_data)}")
        output_parts.append("\n💡 Podes perguntar-me sobre um tipo de local específico para pesquisa detalhada.")

        return "\n".join(output_parts)

    except Exception as e:
        logger.error(f"Error in get_place_categories: {e}")
        return f"❌ Error getting place categories: {str(e)}"


@tool
def search_lisbon_knowledge(
    query: str,
    max_results: int = 5
) -> str:
    """
    Search across ALL Lisbon knowledge bases using semantic search.

    This is the most comprehensive search tool. It searches the PDF guide,
    places, and events databases simultaneously using AI-powered semantic search.

    Args:
        query (str): Natural language query. Examples:
            'What is the Lisboa Card?', 'Best viewpoints in Lisbon',
            'Traditional Portuguese restaurants', 'Public transport tips'.
        max_results (int): Maximum results per source (default: 5).

    Returns:
        str: Combined results from PDF guide, places, and events.

    Examples:
        - search_lisbon_knowledge("Lisboa Card benefits")
        - search_lisbon_knowledge("getting from airport to center")
        - search_lisbon_knowledge("best places to eat pasteis de nata")
    """
    try:
        query = str(query).strip() if query else "Lisbon tourism information"
        if not isinstance(max_results, int) or max_results <= 0:
            max_results = 5

        logger.info(f"search_lisbon_knowledge: query='{query}', max={max_results}")

        kb = _get_vector_store()

        if not kb:
            return "❌ Vector store not available. Try search_cultural_events or search_places_attractions instead."

        focus = _infer_lisbon_knowledge_focus(query)
        per_source_limit = max(1, min(4, max_results))
        candidate_limit = max(per_source_limit + 1, 4)

        try:
            pdf_results = kb.search_with_scores(
                query=query,
                k=candidate_limit,
                collections=[COLLECTION_PDF],
            )
            places_results = kb.search_with_scores(
                query=query,
                k=candidate_limit,
                collections=[COLLECTION_PLACES],
            )
            events_results = kb.search_with_scores(
                query=query,
                k=candidate_limit,
                collections=[COLLECTION_EVENTS],
            )
        except Exception as e:
            logger.error(f"search_lisbon_knowledge search error: {e}")
            return f"❌ Search failed: {str(e)}. Try search_cultural_events or search_places_attractions instead."

        if not pdf_results and not places_results and not events_results:
            return f"No results found for: '{query}'"

        output_parts = ["🔍 **Lisbon Knowledge Search Results:**\n"]
        output_parts.append(f"Query: \"{query}\"\n")

        ordered_sections = {
            "guide": [
                ("📚", "Guide / PDF Knowledge", pdf_results),
                ("🏛️", "Related Places", places_results),
                ("📅", "Related Events", events_results),
            ],
            "places": [
                ("🏛️", "Related Places", places_results),
                ("📚", "Guide / PDF Knowledge", pdf_results),
                ("📅", "Related Events", events_results),
            ],
            "events": [
                ("📅", "Related Events", events_results),
                ("🏛️", "Related Places", places_results),
                ("📚", "Guide / PDF Knowledge", pdf_results),
            ],
            "general": [
                ("📚", "Guide / PDF Knowledge", pdf_results),
                ("🏛️", "Related Places", places_results),
                ("📅", "Related Events", events_results),
            ],
        }[focus]

        shown_count = 0
        for icon, heading, scored_results in ordered_sections:
            if not scored_results:
                continue

            output_parts.append(f"\n{icon} **{heading}:**")
            for doc, _score in scored_results[:per_source_limit]:
                title = str(doc.metadata.get('title', 'Unknown')).strip()
                snippet = _extract_knowledge_doc_snippet(doc)
                if snippet and title and title.lower() not in snippet.lower():
                    output_parts.append(f"   • **{title}** — {snippet}")
                elif snippet:
                    output_parts.append(f"   • {snippet}")
                else:
                    output_parts.append(f"   • **{title}**")
                shown_count += 1

        output_parts.append(
            f"\n\n📊 **Result slices shown:** {shown_count} "
            f"({min(len(pdf_results), per_source_limit)} guide, {min(len(places_results), per_source_limit)} places, {min(len(events_results), per_source_limit)} events)"
        )

        return "\n".join(output_parts)

    except Exception as e:
        logger.error(f"Error in search_lisbon_knowledge: {e}")
        return f"❌ Error searching knowledge base: {str(e)}"


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("\033[1m🧪 COMPREHENSIVE TEST: VisitLisboa Semantic Search Tools\033[0m")
    print("=" * 70)

    test_results = {"passed": 0, "failed": 0, "total": 0}

    def run_test(test_name: str, test_func, *args, **kwargs):
        """Helper to run tests with error handling."""
        test_results["total"] += 1
        print(f"\n\033[1m{'─' * 70}\033[0m")
        print(f"\033[1;36m🔬 TEST {test_results['total']}: {test_name}\033[0m")
        print(f"{'─' * 70}")
        try:
            result = test_func(*args, **kwargs)
            print(result)
            test_results["passed"] += 1
            print("\n\033[1;32m✅ PASSED\033[0m")
            return result
        except Exception as e:
            print(f"\n\033[1;31m❌ FAILED: {str(e)}\033[0m")
            test_results["failed"] += 1
            return None

    # =========================================================================
    # CATEGORY DISCOVERY TESTS
    # =========================================================================
    # TEST 1: Get Event Categories
    run_test(
        "Get Event Categories",
        get_event_categories.invoke,
        {}
    )
    # TEST 2: Get Place Categories
    run_test(
        "Get Place Categories",
        get_place_categories.invoke,
        {}
    )

    # =========================================================================
    # EVENT SEARCH TESTS (with different date filters)
    # CRITICAL: Verifying TEMPORAL RELEVANCE - ephemeral events should rank first!
    # =========================================================================
    # TEST 3: Search Events - Semantic Query (Music)
    # .NOTE: Should find concerts/music events via token matching + synonyms
    run_test(
        "Search Events - Semantic Query (Music) [TESTS TOKEN MATCHING]",
        search_cultural_events.invoke,
        {"query": "music concerts", "max_results": 3}
    )

    # TEST 4: Search Events - By Category (Exhibitions)
    # .NOTE: Even exhibitions should be sorted by temporal relevance
    run_test(
        "Search Events - By Category (Exhibitions) [TESTS DURATION SORTING]",
        search_cultural_events.invoke,
        {"category": "Exhibitions", "max_results": 3}
    )

    # TEST 5: Search Events - This Week
    # .CRITICAL: Single-day events should appear BEFORE long exhibitions!
    run_test(
        "Search Events - This Week [TESTS TEMPORAL RELEVANCE]",
        search_cultural_events.invoke,
        {"date_filter": "this week", "max_results": 5}
    )

    # TEST 6: Search Events - This Weekend
    run_test(
        "Search Events - This Weekend [TESTS TEMPORAL RELEVANCE]",
        search_cultural_events.invoke,
        {"date_filter": "this weekend", "max_results": 5}
    )

    # TEST 7: Search Events - Next Month
    run_test(
        "Search Events - Next Month",
        search_cultural_events.invoke,
        {"date_filter": "next month", "max_results": 3}
    )

    # TEST 8: Search Events - Today
    # CRITICAL: Only TODAY events, single-day performances prioritized
    run_test(
        "Search Events - Today [CRITICAL: EPHEMERAL FIRST]",
        search_cultural_events.invoke,
        {"date_filter": "today", "max_results": 5}
    )

    # =========================================================================
    # PLACE SEARCH TESTS (including Dados Abertos fallback)
    # =========================================================================
    # TEST 9: Search Places - Semantic Query
    run_test(
        "Search Places - Semantic Query (Museums)",
        search_places_attractions.invoke,
        {"query": "historic museums art", "max_results": 3}
    )

    # TEST 10: Search Places - By Category
    run_test(
        "Search Places - By Category (Restaurants)",
        search_places_attractions.invoke,
        {"category": "Restaurant", "max_results": 3}
    )

    # TEST 11: Search Places - View Points
    run_test(
        "Search Places - View Points",
        search_places_attractions.invoke,
        {"query": "panoramic views sunset", "max_results": 3}
    )

    # TEST 12: HYBRID SEARCH - Hospital (VisitLisboa + Dados Abertos)
    # .NOTE: "hospital" keyword triggers hybrid mode, combining tourist data
    # with public infrastructure from Lisboa Aberta (GPS coords included)
    run_test(
        "HYBRID SEARCH - Hospital (VisitLisboa + Dados Abertos)",
        search_places_attractions.invoke,
        {"query": "hospital", "max_results": 5}
    )

    # TEST 13: HYBRID SEARCH - University/Education
    # .NOTE: "universidade" keyword triggers Dados Abertos for public institutions
    run_test(
        "HYBRID SEARCH - University (VisitLisboa + Dados Abertos)",
        search_places_attractions.invoke,
        {"query": "universidade", "max_results": 5}
    )

    # TEST 14: Tourist Query (NO hybrid, VisitLisboa only)
    # .NOTE: "tower belem" is a tourist query, should NOT trigger Dados Abertos
    run_test(
        "Tourist Query - Torre de Belém (VisitLisboa only)",
        search_places_attractions.invoke,
        {"query": "tower belem monument", "max_results": 3}
    )

    # =========================================================================
    # COMPREHENSIVE KNOWLEDGE SEARCH TESTS
    # =========================================================================
    # TEST 15: Search Knowledge - Lisboa Card Info
    run_test(
        "Search Knowledge - Lisboa Card Info",
        search_lisbon_knowledge.invoke,
        {"query": "Lisboa Card benefits", "max_results": 3}
    )

    # TEST 16: Search Knowledge - Transport Tips
    run_test(
        "Search Knowledge - Transport Tips",
        search_lisbon_knowledge.invoke,
        {"query": "public transport metro tram", "max_results": 3}
    )

    # TEST 17: Search Knowledge - Food & Gastronomy
    run_test(
        "Search Knowledge - Food & Gastronomy",
        search_lisbon_knowledge.invoke,
        {"query": "pasteis de nata traditional food", "max_results": 3}
    )

    # =========================================================================
    # EDGE CASES & ERROR HANDLING
    # =========================================================================

    # TEST 18: Edge Case - Empty Query
    run_test(
        "Edge Case - Empty Query (Events)",
        search_cultural_events.invoke,
        {"max_results": 3}
    )

    # TEST 19: Edge Case - Empty Query
    run_test(
        "Edge Case - Empty Query (Places)",
        search_places_attractions.invoke,
        {"max_results": 3}
    )

    # TEST 20: Edge Case - Invalid Date Format
    run_test(
        "Edge Case - Invalid Date Format",
        search_cultural_events.invoke,
        {"date_filter": "invalid_date_xyz", "max_results": 3}
    )

    # =========================================================================
    # TEST SUMMARY
    # =========================================================================

    print("\n" + "=" * 70)
    print("\033[1m📊 TEST SUMMARY\033[0m")
    print("=" * 70)
    print(f"\033[1;32m✅ Passed: {test_results['passed']}/{test_results['total']}\033[0m")
    print(f"\033[1;31m❌ Failed: {test_results['failed']}/{test_results['total']}\033[0m")

    if test_results['failed'] == 0:
        print("\n\033[1;32m🎉 ALL TESTS PASSED! System is working correctly.\033[0m")
    else:
        print("\n\033[1;33m⚠️  Some tests failed. Check errors above.\033[0m")

    print("=" * 70 + "\n")
