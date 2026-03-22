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

import io
import json
import logging
import math
import os
import re
import unicodedata
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
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
    raw = (date_filter or "upcoming").strip()
    if language != "pt":
        return raw

    mapping = {
        "today": "hoje",
        "tomorrow": "amanhã",
        "this week": "esta semana",
        "next week": "próxima semana",
        "this weekend": "este fim de semana",
        "this month": "este mês",
        "next month": "próximo mês",
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
    if start_date and end_date:
        connector = "a" if language == "pt" else "to"
        date_window = f"{start_date.strftime('%Y-%m-%d')} {connector} {(end_date - timedelta(days=1)).strftime('%Y-%m-%d')}"
    elif start_date:
        date_window = start_date.strftime('%Y-%m-%d')
    else:
        date_window = "open range"

    normalized_query = (query or "").strip()
    normalized_category = (category or "").strip()
    normalized_filter = _localize_event_date_filter(date_filter, language=language)
    normalized_category = _localize_event_category(normalized_category, language=language)
    shown_from = offset + 1 if shown_results > 0 else 0
    shown_to = offset + shown_results if shown_results > 0 else 0

    if language == "pt":
        scope_parts = [f"{normalized_filter} ({date_window})"]
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

    scope_parts = [f"{normalized_filter} ({date_window})"]
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


_EVENT_GENERIC_QUERY_TERMS = {
    'event', 'events', 'evento', 'eventos',
    'cultura', 'cultural', 'culture', 'culturais',
    'lisbon', 'lisboa', 'portugal', 'city', 'cidade',
    'great', 'major', 'grandes', 'explorar', 'explore',
    'find', 'finding', 'search', 'show', 'mostrar', 'mostra', 'encontra', 'encontre',
    'procura', 'procure', 'descobre', 'discover', 'want', 'quero',
    'this', 'week', 'esta', 'semana', 'what', 'which', 'que', 'quais',
    'there', 'happening', 'temos', 'have', 'local', 'locais',
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
}


def _expand_event_query_tokens(query: Optional[str]) -> List[str]:
    """Builds a small expanded token set for event text matching."""
    if not query:
        return []

    normalized_query = _normalize_place_hint_text(query)
    normalized_query = re.sub(r"\bmusica\s+ao\s+vivo\b", "live music", normalized_query)
    normalized_query = re.sub(r"\bao\s+vivo\b", "live", normalized_query)
    original_tokens = [t.strip().lower() for t in normalized_query.split() if len(t.strip()) >= 3]
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

    searchable = " ".join([
        event.get('title', ''),
        event.get('full_description', ''),
        event.get('short_description', ''),
        event.get('category', ''),
    ]).lower()
    return any(token in searchable for token in expanded_tokens)


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
    'farmacia', 'farmácia', 'bombeiros', 'policia', 'polícia', 'segurança', 'seguranca',
    # Education
    'escola', 'colegio', 'colégio', 'universidade', 'faculdade', 'instituto', 'creche',
    # Culture (complement)
    'biblioteca', 'teatro', 'cinema', 'galeria', 'monumento', 'miradouro', 'igreja',
    # Outdoors & Leisure
    'jardim', 'parque', 'piscina', 'desporto',
    # Services & Amenities
    'wc', 'banheiro', 'sanitário', 'sanitario', 'estacionamento', 'parking',
    'embaixada', 'cemiterio', 'cemitério', 'junta', 'câmara', 'camara',
    # Streets & Locations
    'rua', 'avenida', 'praça', 'praca', 'largo', 'bairro'
}


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
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in DADOS_ABERTOS_KEYWORDS)


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

    for token in tokens:
        if token in name_text:
            score += 3.0
        elif token in address_text:
            score += 1.5
        elif token in combined_text:
            score += 1.0

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
            _search_place_in_datasets_logic,
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
    normalized_query = _normalize_place_hint_text(query)
    return any(token in normalized_query for token in ["best", "top", "recommended", "recommend", "must-see", "must see"])


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

    tokens = re.findall(r"[a-z0-9]+", _normalize_place_hint_text(query))
    return [
        token for token in tokens
        if len(token) >= 3 and token not in _GENERIC_PLACE_QUERY_TOKENS
    ]


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


def _extract_required_service_term_groups(query: Optional[str]) -> List[set[str]]:
    """Extracts critical service-intent term groups that must remain present in results."""
    normalized_query = _normalize_place_hint_text(query)
    query_tokens = set(re.findall(r"[a-z0-9]+", normalized_query))
    required_groups: List[set[str]] = []
    for variants in _PUBLIC_SERVICE_FOCUS_TERMS.values():
        if any(
            (_normalize_place_hint_text(term) in normalized_query if " " in term else _normalize_place_hint_text(term) in query_tokens)
            for term in variants
        ):
            required_groups.append(variants)
    return required_groups


def _matches_required_service_terms(searchable_text: str, required_groups: List[set[str]]) -> bool:
    """Checks whether candidate text preserves the critical service intent from the query."""
    if not required_groups:
        return True

    normalized_text = _normalize_place_hint_text(searchable_text)
    text_tokens = set(re.findall(r"[a-z0-9]+", normalized_text))
    return any(
        any(
            (_normalize_place_hint_text(term) in normalized_text if " " in term else _normalize_place_hint_text(term) in text_tokens)
            for term in variants
        )
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
    query_lower = (query or "").lower()
    museum_terms = ["museum", "museu", "museums", "museus"]
    monument_terms = ["monument", "monumento", "monuments", "monumentos", "monastery", "castle", "palace", "church"]
    top_attraction_terms = [
        "atrações imperdíveis", "atracoes imperdiveis", "must-see", "must see",
        "first time", "primeira vez", "top attractions", "main attractions",
        "highly recommended attractions", "o que visitar",
    ]

    museum_requested = any(term in query_lower for term in museum_terms)
    monument_requested = any(term in query_lower for term in monument_terms)
    top_attractions_requested = any(term in query_lower for term in top_attraction_terms)

    if top_attractions_requested:
        return "top_attractions"

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
    if any(term in query_lower for term in ["restaurant", "restaurante", "food", "dinner", "lunch", "brunch", "gastronomy"]):
        return "food"
    if any(term in query_lower for term in ["hotel", "stay", "accommodation", "guest house"]):
        return "accommodation"
    return None


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
    
    for item in data:
        # Category filter
        if category and not _place_category_matches(item.get('category', ''), category):
            continue
        
        searchable = " ".join([
            item.get('title', ''),
            item.get('full_description', ''),
            item.get('short_description', ''),
            item.get('category', ''),
            item.get('location', '')
        ]).lower()
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
                if not any(token in normalized_searchable for token in query_tokens):
                    continue
            elif query_lower not in searchable:
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

        # Parse date range (CRITICAL: defaults to upcoming 30 days if not specified)
        if not date_filter:
            date_filter = 'upcoming'  # Default to next 30 days
        
        start_date, end_date = parse_date_range(date_filter)
        
        # Logging
        date_info = f"{start_date.strftime('%Y-%m-%d') if start_date else 'any'} to {end_date.strftime('%Y-%m-%d') if end_date else 'any'}"
        logger.info(
            f"search_cultural_events: query='{query}', category='{category}', dates={date_info}, max={max_results}, offset={offset}"
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
        if query:
            expanded_tokens = _expand_event_query_tokens(query)

            # If we have meaningful tokens left, apply the filter
            if expanded_tokens:
                events_data = [event for event in events_data if _event_matches_query(event, expanded_tokens)]
                undated_candidates = [event for event in undated_candidates if _event_matches_query(event, expanded_tokens)]
                logger.info(f"After query filter: {len(events_data)} events (tokens: {expanded_tokens[:5]}...)")
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
        
        logger.info(
            f"search_places_attractions: query='{query}', category='{category}', max={max_results}, offset={offset}"
        )
        required_service_terms = _extract_required_service_term_groups(query)
        
        # Check if we should also search Dados Abertos (hybrid mode)
        search_dados_abertos = _should_search_dados_abertos(query)
        dados_abertos_results = []
        
        if search_dados_abertos and query:
            logger.info("Hybrid mode: Query contains Dados Abertos keywords")
            dados_abertos_results = _search_dados_abertos_hybrid(query, max_results=requested_window // 2 + 1)
            logger.info(f"Dados Abertos returned {len(dados_abertos_results)} results")
        
        # =====================================================================
        # STEP 1: Search VisitLisboa (Vector Store)
        # =====================================================================
        visitlisboa_results = []
        kb = _get_vector_store()
        
        if kb:
            try:
                query_intent = _infer_place_query_intent(query, category)
                requested_category = _normalize_place_category_filter(category)
                search_query = query or "places and attractions in Lisbon"
                if query_intent == "top_attractions":
                    search_query = f"iconic attractions monuments viewpoints historic sites {search_query}"
                if category:
                    category_prefix = category
                    if requested_category == "museums & monuments" and query_intent == "museum_only":
                        category_prefix = "Museums"
                    search_query = f"{category_prefix} {search_query}"

                query_tokens = _extract_place_query_tokens(query or search_query)
                location_hints = _extract_place_location_hints(query or search_query)
                ranking_requested = _query_requests_ranked_places(query or search_query) or query_intent == "top_attractions"
                
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

                    token_hits = sum(1 for token in query_tokens if token in normalized_searchable)
                    title_hits = sum(1 for token in query_tokens if token in normalized_title)

                    if query_tokens and token_hits == 0 and vector_score > 1.25:
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
                    title_bonus = min(0.18, title_hits * 0.09)
                    token_bonus = min(0.15, token_hits * 0.05)
                    service_penalty = 0.25 if query_tokens and _is_service_like_place_category(item_category) and query_intent != "accommodation" else 0.0
                    museum_specific_bonus = 0.10 if query_intent == "museum_only" and _is_explicit_museum_candidate(
                        metadata.get('title', ''),
                        metadata.get('url', ''),
                        searchable,
                    ) else 0.0
                    top_attraction_bonus = _top_attraction_category_bonus(item_category) if query_intent == "top_attractions" else 0.0
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
                        + top_attraction_bonus
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
        if query and len(visitlisboa_results) < requested_window:
            fallback_items = _fallback_search(
                query=query,
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
            if query:
                fallback_items = _fallback_search(query, category, _load_places_json(), max_results=requested_window)
                if fallback_items:
                    all_results = [_convert_raw_place_to_result(item) for item in fallback_items]

            if not all_results and query:
                logger.info("No results from hybrid search, trying direct Dados Abertos")
                from tools.dados_abertos import _search_place_in_datasets_logic
                open_data_results = _search_place_in_datasets_logic(query, max_results=requested_window)
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
            
            # Lisboa Card discount (from enriched data)
            if full_data and full_data.get('lisboa_card_discount'):
                output_parts.append(f"   🎫 {full_data['lisboa_card_discount']}")
            
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
                for schedule in full_data['schedules']:
                    if schedule.get('today'):
                        output_parts.append(f"   🕐 {schedule['today']}")
                        break
            
            # Tickets/prices (from enriched data)
            if full_data and full_data.get('tickets_offers'):
                tickets = full_data['tickets_offers']
                if tickets.get('description'):
                    price_desc = tickets['description'][:80]
                    if len(tickets['description']) > 80:
                        price_desc += "..."
                    output_parts.append(f"   💰 {price_desc}")
            
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
        
        logger.info(f"search_lisbon_knowledge: query='{query}', max={max_results}")
        
        kb = _get_vector_store()
        
        if not kb:
            return "❌ Vector store not available. Try search_cultural_events or search_places_attractions instead."
        
        try:
            # Search all collections
            results = kb.search(query=query, k=max_results)
        except Exception as e:
            logger.error(f"search_lisbon_knowledge search error: {e}")
            return f"❌ Search failed: {str(e)}. Try search_cultural_events or search_places_attractions instead."
        
        if not results:
            return f"No results found for: '{query}'"
        
        # Group by source
        pdf_results = []
        places_results = []
        events_results = []
        
        for doc in results:
            source = doc.metadata.get('source', 'unknown').lower()
            if 'pdf' in source or 'guide' in source or 'turismo' in source:
                pdf_results.append(doc)
            elif 'place' in source:
                places_results.append(doc)
            elif 'event' in source:
                events_results.append(doc)
        
        output_parts = ["🔍 **Lisbon Knowledge Search Results:**\n"]
        output_parts.append(f"Query: \"{query}\"\n")
        
        if pdf_results:
            output_parts.append("\n📚 **From Lisboa Card Guide:**")
            for doc in pdf_results[:3]:
                content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                output_parts.append(f"   • {content}")
        
        if places_results:
            output_parts.append("\n🏛️ **Related Places:**")
            for doc in places_results[:3]:
                title = doc.metadata.get('title', 'Unknown')
                output_parts.append(f"   • {title}")
        
        if events_results:
            output_parts.append("\n📅 **Related Events:**")
            for doc in events_results[:3]:
                title = doc.metadata.get('title', 'Unknown')
                output_parts.append(f"   • {title}")
        
        output_parts.append(f"\n\n📊 **Total results:** {len(results)}")
        
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
