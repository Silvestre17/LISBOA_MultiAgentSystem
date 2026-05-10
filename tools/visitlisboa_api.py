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
#   Usage:
#     > python tools/visitlisboa_api.py
#       Run the manual VisitLisboa semantic-search tool suite against the local vector store and JSON fallbacks.
#
#   Note: Requires vector store to be built first with vector_store.py
# ==========================================================================

# Required libraries:
# pip install langchain-core langchain-chroma langchain-huggingface

import contextlib
import json
import logging
import math
import os
import re
import threading
import unicodedata
import warnings
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

from langchain_core.tools import tool

# Suppress chromadb telemetry warnings
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

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
MAX_USER_FACING_RESULTS = 5


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
                with contextlib.suppress(ValueError):
                    dates.append(datetime.strptime(start_iso, '%Y-%m-%d'))
            if end_iso:
                with contextlib.suppress(ValueError):
                    dates.append(datetime.strptime(end_iso, '%Y-%m-%d'))

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
                display = _localize_event_date_text(display, language=language)
                if time:
                    connector = "às" if language == "pt" else "at"
                    formatted.append(f"{display} {connector} {time}")
                else:
                    formatted.append(display)
        elif date_entry.get('type') == 'range':
            start = date_entry.get('start', {}).get('display_text', '')
            end = date_entry.get('end', {}).get('display_text', '')
            if start and end:
                start = _localize_event_date_text(start, language=language)
                end = _localize_event_date_text(end, language=language)
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


_EVENT_MONTHS_PT = {
    "jan": "janeiro",
    "january": "janeiro",
    "feb": "fevereiro",
    "february": "fevereiro",
    "mar": "março",
    "march": "março",
    "apr": "abril",
    "april": "abril",
    "may": "maio",
    "jun": "junho",
    "june": "junho",
    "jul": "julho",
    "july": "julho",
    "aug": "agosto",
    "august": "agosto",
    "sep": "setembro",
    "sept": "setembro",
    "september": "setembro",
    "oct": "outubro",
    "october": "outubro",
    "nov": "novembro",
    "november": "novembro",
    "dec": "dezembro",
    "december": "dezembro",
}


def _localize_event_title(title: Optional[str], language: str = "en") -> str:
    """Localize known scraped VisitLisboa event titles for PT-PT responses."""
    cleaned = (title or "").strip()
    if language != "pt" or not cleaned:
        return cleaned

    cleaned = re.sub(
        r"\bBook\s+Fair'?\s*(?P<year>\d{2})\b",
        lambda match: f"Feira do Livro 20{match.group('year')}",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\bBook\s+Fair\b", "Feira do Livro", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _localize_event_date_text(value: str, language: str = "en") -> str:
    """Translate simple English VisitLisboa date snippets to PT-PT."""
    if language != "pt" or not value:
        return value

    def replace_date(match: re.Match[str]) -> str:
        day = match.group("day")
        month = _EVENT_MONTHS_PT.get(match.group("month").lower(), match.group("month"))
        year = match.group("year")
        return f"{day} de {month} de {year}" if year else f"{day} de {month}"

    localized = re.sub(
        r"(?P<day>\d{1,2})\s+(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?),?\s*(?P<year>\d{4})?",
        replace_date,
        value,
        flags=re.IGNORECASE,
    )
    localized = re.sub(r"\bto\b", "a", localized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", localized).strip()


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
        return re.sub(r"\bPaid\b", "Pago", localized, flags=re.IGNORECASE)

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


def _localize_place_category(category: Optional[str], language: str = "en") -> str:
    """Localizes common VisitLisboa place categories."""
    raw = (category or "").strip()
    if not raw or language != "pt":
        return raw
    mapping = {
        "General": "Geral",
        "Museums & Monuments": "Museus e Monumentos",
        "Museums": "Museus",
        "Monuments": "Monumentos",
        "Restaurants": "Restaurantes",
        "Restaurant": "Restaurante",
        "Attractions": "Atrações",
        "Attraction": "Atração",
        "Hotels": "Hotéis",
        "View Points": "Miradouros",
        "Parks & Gardens": "Parques e Jardins",
        "Shopping": "Compras",
        "Nightlife": "Vida noturna",
        "Tours": "Tours",
        "Beaches": "Praias",
    }
    return mapping.get(raw, raw)


def _localize_place_title(title: Optional[str], language: str = "en") -> str:
    """Localize known VisitLisboa place titles for PT-PT output."""
    raw = (title or "").strip()
    if not raw or language != "pt":
        return raw
    mapping = {
        "Castle Museum": "Castelo de São Jorge",
        "Castle of the Moors": "Castelo dos Mouros",
        "Carmo Archaeological Museum": "Museu Arqueológico do Carmo",
        "Fronteira Palace": "Palácio Fronteira",
        "João de Deus Museum": "Museu João de Deus",
        "Joao de Deus Museum": "Museu João de Deus",
        "Words Factory": "Fábrica das Palavras",
        "Monument to the Discoveries": "Padrão dos Descobrimentos",
        "National Museum of Natural History and Science": "Museu Nacional de História Natural e da Ciência",
        "National Palace and Gardens of Queluz": "Palácio Nacional e Jardins de Queluz",
        "National Tile Museum": "Museu Nacional do Azulejo",
        "Pena National Palace": "Palácio Nacional da Pena",
        "Palace of Belém": "Palácio de Belém",
        "Palace of Belem": "Palácio de Belém",
        "Prazeres Cemetery and Museum": "Cemitério e Museu dos Prazeres",
        "Museum of Aljube – Resistance and Freedom": "Museu do Aljube - Resistência e Liberdade",
        "Museum of Aljube - Resistance and Freedom": "Museu do Aljube - Resistência e Liberdade",
        "Museum of the Lisbon Geographical Society": "Museu da Sociedade de Geografia de Lisboa",
        "Roman Galleries": "Galerias Romanas",
        "Jerónimos Monastery": "Mosteiro dos Jerónimos",
        "Jeronimos Monastery": "Mosteiro dos Jerónimos",
        "Combatant's Museum in Forte do Bom Sucesso": "Museu dos Combatentes no Forte do Bom Sucesso",
    }
    if raw in mapping:
        return mapping[raw]
    generic_replacements = [
        (r"^Atelier-Museum\s+(.+)$", r"Atelier-Museu \1"),
        (r"^House-Museum\s+(.+)$", r"Casa-Museu \1"),
        (r"^Museum of Lisbon\s+(.+)$", r"Museu de Lisboa - \1"),
        (r"^Museum of the\s+(.+)$", r"Museu da \1"),
        (r"^Museum of\s+(.+)$", r"Museu de \1"),
    ]
    localized = raw
    for pattern, replacement in generic_replacements:
        localized = re.sub(pattern, replacement, localized)
    return localized


def _known_place_description(title: str, language: str = "en") -> str:
    """Return concise descriptions for high-profile landmarks when scraped text is unavailable."""
    normalized_title = _normalize_lookup_text(title)
    if language == "pt":
        descriptions = {
            "mosteiro dos jeronimos": "Mosteiro manuelino em Belém, classificado como Património Mundial da UNESCO e associado à memória dos Descobrimentos portugueses.",
            "padrao dos descobrimentos": "Monumento ribeirinho em Belém dedicado às figuras históricas ligadas aos Descobrimentos portugueses.",
            "fabrica das palavras": "Biblioteca municipal e espaço cultural junto ao Tejo, projetado por Miguel Arruda e ligado à leitura, arquitetura e programação cultural.",
            "museu joao de deus": "Museu dedicado ao poeta e pedagogo João de Deus, com acervo sobre leitura, educação e métodos históricos de ensino.",
        }
    else:
        descriptions = {
            "mosteiro dos jeronimos": "Manueline monastery in Belém, listed as a UNESCO World Heritage Site and linked to Portugal's Age of Discoveries.",
            "jeronimos monastery": "Manueline monastery in Belém, listed as a UNESCO World Heritage Site and linked to Portugal's Age of Discoveries.",
            "padrao dos descobrimentos": "Riverside monument in Belém dedicated to historical figures associated with the Portuguese Discoveries.",
            "monument to the discoveries": "Riverside monument in Belém dedicated to historical figures associated with the Portuguese Discoveries.",
            "fabrica das palavras": "Municipal library and cultural venue by the Tagus, designed by Miguel Arruda and linked to reading, architecture, and cultural programming.",
            "words factory": "Municipal library and cultural venue by the Tagus, designed by Miguel Arruda and linked to reading, architecture, and cultural programming.",
            "museu joao de deus": "Museum devoted to poet and educator João de Deus, with collections about reading, education, and historical teaching methods.",
            "joao de deus museum": "Museum devoted to poet and educator João de Deus, with collections about reading, education, and historical teaching methods.",
        }
    return descriptions.get(normalized_title, "")


def _localize_visitlisboa_description(
    description: Optional[str],
    language: str = "en",
) -> str:
    """Avoid leaking raw English scraped descriptions into PT-PT answers."""
    raw = re.sub(r"\s+", " ", (description or "").strip())
    if not raw or language != "pt":
        return raw

    english_markers = [
        " the ",
        " and ",
        " with ",
        " from ",
        " here",
        " world ",
        " visitors ",
        " experience",
        " innovation",
        " discover",
        " located",
        " offers ",
        " across ",
        " clothing ",
        " second life",
        " recycling ",
        " event that ",
    ]
    padded = f" {raw.lower()} "
    if any(marker in padded for marker in english_markers):
        return ""
    return raw


def _localize_place_value_text(value: Optional[str], language: str = "en") -> str:
    """Localizes common VisitLisboa place field values for PT-PT output."""
    raw = re.sub(r"\s+", " ", (value or "").strip())
    if not _clean_user_facing_value(raw):
        return ""
    if language != "pt":
        cleaned = re.sub(r"^Price\s*:\s*", "", raw, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bGratis\b", "Free", cleaned, flags=re.IGNORECASE)
        return _clean_user_facing_value(cleaned)

    localized = raw
    localized = re.sub(r"^Price\s*:", "Preço:", localized, flags=re.IGNORECASE)
    localized = re.sub(
        r"\bChildren\s+(?:Free|Gratis|Gratuito)\s+until\s*(?:\(age\)|age)?\s*:?\s*(\d+)",
        r"Crianças grátis até aos \1 anos",
        localized,
        flags=re.IGNORECASE,
    )
    localized = re.sub(r"\bFree\s+with\s+Lisboa\s+Card\b", "Gratuito com Lisboa Card", localized, flags=re.IGNORECASE)
    localized = re.sub(r"\bwith\s+Lisboa\s+Card\b", "com Lisboa Card", localized, flags=re.IGNORECASE)
    localized = re.sub(r"\bChildren\s*:", "Crianças:", localized, flags=re.IGNORECASE)
    localized = re.sub(r"\bGratis\b", "Gratuito", localized, flags=re.IGNORECASE)
    localized = re.sub(r"\bAdult\s*:", "Adulto:", localized, flags=re.IGNORECASE)
    localized = re.sub(r"\bSenior(\s*\([^)]*\))?\s*:", r"Sénior\1:", localized, flags=re.IGNORECASE)
    localized = re.sub(r"\bFree\b", "Gratuito", localized, flags=re.IGNORECASE)
    localized = re.sub(r"\+\s*info\b", "", localized, flags=re.IGNORECASE).strip(" ;,.")
    return _clean_user_facing_value(localized)


def _compact_place_ticket_price_text(value: Optional[str], language: str = "en", max_chars: int = 115) -> str:
    """Return a compact user-facing summary of scraped VisitLisboa ticket prices."""
    cleaned = re.sub(r"\s+", " ", (value or "").strip())
    if not cleaned:
        return ""

    cleaned = re.sub(r"^(?:link|links)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^Price\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    if language != "pt":
        cleaned = re.sub(r"\bGratis\b", "Free", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\bChildren\s+Free\s+until\s*\(age\)\s*:\s*(\d+)",
        r"Children free until age \1",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?=(?:Children(?:\s*\([^)]*\))?|Adult|Adults|Family|Senior|Seniors|Student|Students)\s*:)",
        "; ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*;\s*", "; ", cleaned).strip(" ;")
    cleaned = re.sub(r"(?:;\s*){2,}", "; ", cleaned).strip(" ;")

    if len(cleaned) > max_chars:
        parts = [part.strip() for part in cleaned.split(";") if part.strip()]
        compact_parts: List[str] = []
        total_len = 0
        for part in parts:
            projected_len = total_len + len(part) + (2 if compact_parts else 0)
            if projected_len > max_chars:
                break
            compact_parts.append(part)
            total_len = projected_len
        if compact_parts:
            cleaned = "; ".join(compact_parts)
        else:
            cleaned = cleaned[:max_chars].rsplit(" ", 1)[0].strip(" ;,.") + "..."

    return _localize_place_value_text(cleaned, language=language)


def _is_generic_lisbon_location(value: Optional[str]) -> bool:
    """Return whether a VisitLisboa location is only a city-level stub."""
    normalized = _normalize_lookup_text(value)
    return normalized in {"lisboa", "lisbon", "lisboa portugal", "lisbon portugal"}


def _format_visitlisboa_location_line(
    location: Optional[str],
    title: str,
    language: str = "en",
    label_override: Optional[str] = None,
) -> str:
    """Format VisitLisboa location output only when it contains a specific address."""
    loc = re.sub(r"\s+", " ", (location or "").strip())
    known_addresses = {
        "mosteiro dos jeronimos": "Praça do Império, 1400-206 Lisboa",
        "jeronimos monastery": "Praça do Império, 1400-206 Lisboa",
        "padrao dos descobrimentos": "Av. Brasília, 1400-038 Lisboa",
        "monument to the discoveries": "Av. Brasília, 1400-038 Lisboa",
        "museu joao de deus": "Av. Álvares Cabral 69, 1250-017 Lisboa",
        "joao de deus museum": "Av. Álvares Cabral 69, 1250-017 Lisboa",
        "fabrica das palavras": "Largo Mário Magalhães Infante, Cais de Vila Franca de Xira, 2600-187 Vila Franca de Xira",
        "words factory": "Largo Mário Magalhães Infante, Cais de Vila Franca de Xira, 2600-187 Vila Franca de Xira",
    }
    normalized_title = _normalize_lookup_text(title)
    label = label_override or ("Morada" if language == "pt" else "Address")
    if _is_generic_lisbon_location(loc) and normalized_title in known_addresses:
        address = known_addresses[normalized_title]
        query = quote_plus(address)
        return f"    - 📍 **{label}:** [{address}](https://www.google.com/maps/search/?api=1&query={query})"
    if _is_generic_lisbon_location(loc):
        return ""
    if loc:
        query = quote_plus(loc)
        return f"    - 📍 **{label}:** [{loc}](https://www.google.com/maps/search/?api=1&query={query})"
    return ""


def _format_coordinates_location_line(
    lat: Any,
    lon: Any,
    label: str,
    display_text: str = "Map",
) -> str:
    """Format verified coordinates as a map link without exposing raw GPS tuples."""
    try:
        lat_value = float(lat)
        lon_value = float(lon)
    except (TypeError, ValueError):
        return ""
    return (
        f"    - 📍 **{label}:** "
        f"[{display_text}](https://www.google.com/maps/search/?api=1&query={lat_value:.6f},{lon_value:.6f})"
    )


def _clean_user_facing_value(value: Any) -> str:
    """Return a display-safe value, omitting known placeholder strings.

    Args:
        value: Raw scraped value.

    Returns:
        Cleaned text, or an empty string for placeholders.
    """
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    normalized = _normalize_lookup_text(text)
    if normalized in {
        "",
        "n a",
        "na",
        "none",
        "null",
        "not available",
        "nao disponivel",
        "não disponível",
        "indisponivel",
        "indisponível",
        "available soon",
        "check official website",
        "consultar website oficial",
        "info",
        "mais info",
        "more info",
        "buy",
        "tickets",
    }:
        return ""
    if text.strip().lower() in {"+ info", "+info"}:
        return ""
    return text


def _join_user_facing_parts(parts: List[Any]) -> str:
    """Join optional scraped fields without letting ``None`` break search paths."""
    return " ".join(
        clean_part
        for clean_part in (_clean_user_facing_value(part) for part in parts)
        if clean_part
    )


def _is_http_url(value: Any) -> bool:
    """Return whether a value is a safe HTTP(S) URL for Markdown links."""
    text = str(value or "").strip()
    return text.startswith(("https://", "http://"))


def _format_markdown_link(label: str, url: Any) -> str:
    """Build a Markdown link only when the URL is usable."""
    clean_url = str(url or "").strip()
    if not _is_http_url(clean_url):
        return ""
    return f"[{label}]({clean_url})"


def _preserve_source_url(url: Any) -> str:
    """Return a scraped source URL without language rewriting.

    Portuguese collection footers may point to ``/pt-pt/...``, but item-level
    VisitLisboa URLs and external official URLs must keep the scraped URL because
    translated item paths are not guaranteed to exist.
    """
    return str(url or "").strip()


def _format_phone_link(phone: Any) -> str:
    """Format a phone number as a readable tel link when possible."""
    raw = str(phone or "").strip()
    if not raw:
        return ""

    digits = re.sub(r"\D+", "", raw)
    if not digits:
        return raw

    if digits.startswith("351") and len(digits) == 12:
        local = digits[3:]
        display = f"+351 {local[:3]} {local[3:6]} {local[6:]}"
        href = f"tel:+{digits}"
    elif len(digits) == 9:
        display = f"+351 {digits[:3]} {digits[3:6]} {digits[6:]}"
        href = f"tel:+351{digits}"
    else:
        display = raw
        href = f"tel:+{digits}" if raw.startswith("+") else f"tel:{digits}"

    return f"[{display}]({href})"


def _iter_named_http_links(value: Any) -> List[Tuple[str, str]]:
    """Extract unique named HTTP links from VisitLisboa link structures."""
    pairs: List[Tuple[str, str]] = []
    seen: set[str] = set()

    if isinstance(value, dict):
        iterable = value.items()
    elif isinstance(value, list):
        iterable = []
        for item in value:
            if isinstance(item, dict):
                iterable.append((item.get("title") or item.get("text") or "Link", item.get("url")))
    else:
        iterable = []

    for raw_label, raw_url in iterable:
        url = str(raw_url or "").strip()
        if not _is_http_url(url) or url in seen:
            continue
        label = _clean_user_facing_value(raw_label) or "Official website"
        pairs.append((label, url))
        seen.add(url)

    return pairs


def _first_named_http_link(value: Any) -> Tuple[str, str] | None:
    """Return the first usable named HTTP link from a scraped link structure."""
    links = _iter_named_http_links(value)
    return links[0] if links else None


def _is_ticket_http_link(label: Any, url: Any) -> bool:
    """Return whether a scraped link is specific enough to be shown as a ticket link."""
    if not _is_http_url(url):
        return False
    normalized_label = _normalize_lookup_text(label)
    normalized_url = _normalize_lookup_text(url)
    if normalized_label in {"info", "mais info", "more info"}:
        return False
    ticket_markers = {
        "ticket", "tickets", "bilhete", "bilhetes", "buy", "comprar",
        "byblueticket", "bol", "ticketline",
    }
    return any(marker in normalized_label or marker in normalized_url for marker in ticket_markers)


def _first_ticket_http_link(value: Any) -> Tuple[str, str] | None:
    """Return the first usable ticket-specific HTTP link from scraped link structures."""
    for label, url in _iter_named_http_links(value):
        if _is_ticket_http_link(label, url):
            return label, url
    return None


def _localize_visitlisboa_schedule_text(value: Any, language: str = "en") -> str:
    """Localize common VisitLisboa schedule fragments for PT-PT output."""
    text = _clean_user_facing_value(value)
    if not text or language != "pt":
        return text

    replacements = [
        (r"\bToday\b", "Hoje"),
        (r"\bClosed\b", "Fechado"),
        (r"\bSunday\b", "Domingo"),
        (r"\bMonday\b", "Segunda-feira"),
        (r"\bTuesday\b", "Terça-feira"),
        (r"\bWednesday\b", "Quarta-feira"),
        (r"\bThursday\b", "Quinta-feira"),
        (r"\bFriday\b", "Sexta-feira"),
        (r"\bSaturday\b", "Sábado"),
        (r"\bTUE\b", "TER"),
        (r"\bWED\b", "QUA"),
        (r"\bTHU\b", "QUI"),
        (r"\bFRI\b", "SEX"),
        (r"\bSAT\b", "SÁB"),
        (r"\bSUN\b", "DOM"),
        (r"\bMON\b", "SEG"),
        (r"\bto\b", "a"),
    ]
    localized = text
    for pattern, replacement in replacements:
        localized = re.sub(pattern, replacement, localized, flags=re.IGNORECASE)
    return localized


def _localize_place_feature_text(value: Any, language: str = "en") -> str:
    """Localize common VisitLisboa feature values for PT-PT output."""
    text = _clean_user_facing_value(value)
    if not text:
        return ""
    if language != "pt":
        return text

    mapping = {
        "Traditional Portuguese": "Cozinha tradicional portuguesa",
        "Live entertainment / Music": "Entretenimento ao vivo / música",
        "Wi-Fi": "Wi-Fi",
    }
    localized = mapping.get(text, text)
    return re.sub(r"\bto\b", "a", localized, flags=re.IGNORECASE)


def _format_compact_feature_summary(features: Any, language: str = "en", max_items: int = 4) -> str:
    """Build a compact feature summary from VisitLisboa feature lists."""
    if not isinstance(features, list):
        return ""
    cleaned_features = [
        _localize_place_feature_text(feature, language=language)
        for feature in features
    ]
    cleaned_features = [feature for feature in cleaned_features if feature]
    return " · ".join(cleaned_features[:max_items])


def _event_icon_for_category(category: str) -> str:
    """Select a representative event icon from a VisitLisboa category."""
    normalized = _normalize_lookup_text(category)
    if "music" in normalized or "musica" in normalized:
        return "🎵"
    if "exhibition" in normalized or "exposicao" in normalized:
        return "🎨"
    if "theater" in normalized or "teatro" in normalized or "dance" in normalized or "danca" in normalized:
        return "🎭"
    if "cinema" in normalized:
        return "🎬"
    if "sport" in normalized or "desporto" in normalized:
        return "🏃"
    if "fair" in normalized or "feira" in normalized or "festival" in normalized:
        return "🎪"
    if "gastronomy" in normalized or "gastronomia" in normalized:
        return "🍷"
    return "📅"


def _place_icon_for_category(category: str) -> str:
    """Select a representative place icon from a VisitLisboa category."""
    normalized = _normalize_lookup_text(category)
    if "restaurant" in normalized or "restaurante" in normalized or "gastronomy" in normalized:
        return "🍽️"
    if "hotel" in normalized or "alojamento" in normalized:
        return "🛏️"
    if "park" in normalized or "garden" in normalized or "jardim" in normalized or "parque" in normalized:
        return "🌳"
    if "beach" in normalized or "praia" in normalized:
        return "🌊"
    if "shopping" in normalized or "compras" in normalized:
        return "🛍️"
    if "open data" in normalized:
        return "📍"
    return "🏛️"


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
        if duration <= 30:
            return f"📆 {duration} dias"
        return f"🏛️ Longa duração ({duration} dias)"

    if duration == 1:
        return "🎯 Single day"
    if duration <= 30:
        return f"📆 {duration} days"
    return f"🏛️ Long-running ({duration} days)"


def _format_event_filter_summary(
    query: Optional[str],
    category: Optional[str],
    date_filter: Optional[str],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
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
    if language == "pt":
        scope_parts = [normalized_filter if not date_filter else f"{normalized_filter} ({date_window})"]
        scope_parts.append(normalized_category if normalized_category else "todas as categorias")
        if normalized_query:
            scope_parts.append(f"foco temático: {normalized_query}")
        else:
            scope_parts.append("pesquisa geral de eventos")
        return [
            "### 🔵 **Eventos encontrados**",
            f"🧭 **Filtro aplicado:** {', '.join(scope_parts)}.",
        ]

    scope_parts = [normalized_filter if not date_filter else f"{normalized_filter} ({date_window})"]
    scope_parts.append(normalized_category if normalized_category else "all categories")
    if normalized_query:
        scope_parts.append(f"theme focus: {normalized_query}")
    else:
        scope_parts.append("broad event discovery")
    return [
        "### 🔵 **Events Found**",
        f"🧭 **Filter used:** {', '.join(scope_parts)}.",
    ]


_QUOTED_LOOKUP_PATTERN = re.compile(r'"([^"\n]{2,120})"|“([^”\n]{2,120})”')
_GENERIC_LOOKUP_MARKER_PATTERN = re.compile(
    r"\b(?:tell me about|tell me more about|more about|what about|how about|details(?: about| on)?|"
    r"information(?: about| on)?|info(?: about| on)?|about(?: the)?|sobre(?: o| a| os| as)?|"
    r"fala me de|diz me sobre|e do|e da|and what about|and how about)\b"
)
_LEADING_LOOKUP_PREFIX_RE = re.compile(
    r"^(?:"
    r"tell me(?: more)? about|what about|how about|details(?: about| on)?|information(?: about| on)?|"
    r"info(?: about| on)?|about(?: the)?|"
    r"where is|where s|when is|when s|is|are|"
    r"sobre(?: o| a| os| as)?|"
    r"fala me mais sobre(?: o| a| os| as)?|fala me(?: de| do| da| dos| das)?|"
    r"diz me mais sobre(?: o| a| os| as)?|diz me sobre(?: o| a| os| as)?|diz me(?: de| do| da| dos| das)?|"
    r"onde fica(?: o| a| os| as)?|onde e(?: o| a| os| as)?|quando e(?: o| a| os| as)?|"
    r"e do|e da|e dos|e das|e o|e a|e os|e as"
    r")\b\s*"
)
_TRAILING_LOOKUP_SUFFIX_RE = re.compile(
    r"\b(?:wheelchair accessible|accessible|accessibility|open|closed|opening hours|hours|"
    r"cadeira de rodas|acessivel|acessível|aberto|aberta|fechado|fechada|horarios|horários)\b.*$"
)


def _normalize_lookup_text(text: Optional[str]) -> str:
    """Normalizes lookup text for cross-language name matching."""
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


_KNOWN_PLACE_LOOKUP_ALIASES = {
    "jeronimos": "Jerónimos Monastery",
    "jeronimos monastery": "Jerónimos Monastery",
    "mosteiro dos jeronimos": "Jerónimos Monastery",
    "mosteiro jeronimos": "Jerónimos Monastery",
    "gulbenkiam": "Gulbenkian Museum",
    "gulbenkian": "Gulbenkian Museum",
    "ccb": "Centro Cultural de Belém",
    "centro cultural de belem": "Centro Cultural de Belém",
    "maat": "Museu de Arte, Arquitetura e Tecnologia",
    "panteao": "Panteão Nacional",
    "panteao nacional": "Panteão Nacional",
    "padrao dos descobrimentos": "Monument to the Discoveries",
    "monumento aos descobrimentos": "Monument to the Discoveries",
    "monumento dos descobrimentos": "Monument to the Discoveries",
    "torre de belem": "Torre de Belém",
    "tour de belem": "Torre de Belém",
}


def _apply_known_place_lookup_alias(query: Optional[str]) -> Optional[str]:
    """Map common PT/EN aliases and typos to VisitLisboa canonical names."""
    normalized = _normalize_lookup_text(query)
    if not normalized:
        return query
    if normalized in _KNOWN_PLACE_LOOKUP_ALIASES:
        return _KNOWN_PLACE_LOOKUP_ALIASES[normalized]
    for alias, canonical in _KNOWN_PLACE_LOOKUP_ALIASES.items():
        if len(alias) >= 4 and re.search(rf"\b{re.escape(alias)}\b", normalized):
            return canonical
    return query


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


def _extract_prefixed_lookup_phrase(
    query: Optional[str],
    *,
    noise_tokens: set[str],
    max_tokens: int,
    strip_trailing_status_terms: bool = False,
) -> Optional[str]:
    """Extract a named subject after removing leading question/lookup phrasing."""
    if not query:
        return None

    for raw_match in _QUOTED_LOOKUP_PATTERN.findall(query):
        candidate = next((part for part in raw_match if part), "")
        normalized_candidate = _normalize_lookup_text(candidate)
        if normalized_candidate:
            return normalized_candidate

    normalized_query = _normalize_lookup_text(query)
    if not normalized_query:
        return None

    stripped = _LEADING_LOOKUP_PREFIX_RE.sub("", normalized_query).strip()
    if strip_trailing_status_terms and stripped:
        stripped = _TRAILING_LOOKUP_SUFFIX_RE.sub("", stripped).strip()
    if stripped == normalized_query:
        return None

    stripped = re.sub(r"^(?:the|a|an|o|a|os|as)\s+", "", stripped).strip()
    stripped = re.sub(r"\s+(?:please|por favor)$", "", stripped).strip()
    tokens = [token for token in _extract_lookup_tokens(stripped) if token not in noise_tokens]
    if not tokens or len(tokens) > max_tokens:
        return None
    return " ".join(tokens)


def _is_strong_specific_event_match(
    event: Dict[str, Any],
    score: float,
    phrase: Optional[str],
    expanded_tokens: Optional[List[str]] = None,
) -> bool:
    """Return whether a scored event result looks like a true named match."""
    phrase_tokens = [token for token in _extract_lookup_tokens(phrase) if not token.isdigit()]
    if not phrase_tokens:
        return False
    threshold = 120.0 if len(phrase_tokens) >= 3 else 108.0
    if score >= threshold:
        return True

    if expanded_tokens:
        expanded_token_set = {token for token in expanded_tokens if token}
        title_variants = {
            _normalize_lookup_text(event.get("title")),
            _normalize_lookup_text(_clean_event_title(event.get("title"), event.get("url", ""))),
            _normalize_lookup_text(_humanize_visitlisboa_slug(event.get("url", ""))),
        }
        for title in title_variants:
            cleaned_title = _strip_lookup_year_tokens(title)
            title_tokens = [token for token in _extract_lookup_tokens(cleaned_title) if not token.isdigit()]
            if (
                title_tokens
                and len(title_tokens) >= min(2, len(phrase_tokens))
                and len(title_tokens) <= len(phrase_tokens)
                and all(token in expanded_token_set for token in title_tokens)
            ):
                return True

    return False


_PLACE_LOOKUP_SOFT_TOKENS = {"de", "do", "da", "dos", "das", "the", "of", "and"}


def _normalize_specific_place_signature(text: Optional[str]) -> str:
    """Normalize a place name while ignoring lightweight connector tokens."""
    tokens = [
        token for token in _extract_lookup_tokens(text)
        if not token.isdigit() and token not in _PLACE_LOOKUP_SOFT_TOKENS
    ]
    return " ".join(tokens)


def _score_specific_place_lookup_match(place: Dict[str, Any], phrase: Optional[str]) -> float:
    """Score how strongly a place matches a named lookup phrase."""
    normalized_specific = _normalize_lookup_text(phrase)
    if not normalized_specific:
        return 0.0

    searchable = _normalize_lookup_text(_build_place_searchable_text(place))
    title = _normalize_lookup_text(place.get("title"))
    if not searchable and not title:
        return 0.0

    specific_signature = _normalize_specific_place_signature(normalized_specific)
    title_signature = _normalize_specific_place_signature(title)
    searchable_signature = _normalize_specific_place_signature(searchable)

    score = 0.0
    if specific_signature and specific_signature == title_signature:
        score += 140.0
    elif normalized_specific == title:
        score += 140.0
    elif specific_signature and specific_signature in title_signature:
        score += 100.0
    elif normalized_specific in title:
        score += 100.0
    elif specific_signature and specific_signature in searchable_signature:
        score += 60.0
    elif normalized_specific in searchable:
        score += 60.0

    phrase_score = max(
        _phrase_similarity_score(specific_signature or normalized_specific, title_signature or title),
        _phrase_similarity_score(_strip_lookup_year_tokens(specific_signature or normalized_specific), title_signature or title),
    )
    if phrase_score > 0:
        score += 70.0 * phrase_score

    specific_tokens = [token for token in _extract_lookup_tokens(specific_signature or normalized_specific) if not token.isdigit()]
    if specific_tokens:
        title_hits, title_weighted_score = _collect_token_match_stats(specific_tokens, title_signature or title)
        text_hits, text_weighted_score = _collect_token_match_stats(specific_tokens, searchable_signature or searchable)
        if title_hits == len(specific_tokens):
            score += 48.0
        score += min(36.0, title_weighted_score * 10.0)
        score += min(18.0, text_weighted_score * 3.0)

    return score


def _is_strong_specific_place_match(score: float, phrase: Optional[str]) -> bool:
    """Return whether a place score is strong enough to count as a named match."""
    phrase_tokens = [token for token in _extract_lookup_tokens(_normalize_specific_place_signature(phrase) or phrase) if not token.isdigit()]
    if not phrase_tokens:
        return False
    threshold = 118.0 if len(phrase_tokens) >= 3 else 108.0
    return score >= threshold


def _build_specific_lookup_fallback_intro(
    requested_name: str,
    *,
    language: str,
    content_kind: str,
) -> str:
    """Build an explicit intro when an exact named lookup is not found."""
    safe_name = requested_name.strip() or ("esse pedido" if language == "pt" else "that request")
    if language == "pt":
        if content_kind == "event":
            return (
                f"❌ Não encontrei um evento específico com o nome **{safe_name}** na base de dados disponível. "
                "Como alternativa, deixo abaixo eventos do mesmo tipo, estilo ou afinidade temática."
            )
        return (
            f"❌ Não encontrei um local específico com o nome **{safe_name}** na base de dados disponível. "
            "Como alternativa, deixo abaixo locais do mesmo tipo, estilo ou afinidade temática."
        )
    if content_kind == "event":
        return (
            f"❌ I could not find a specific event named **{safe_name}** in the available database. "
            "As an alternative, here are events with a similar type, style, or thematic affinity."
        )
    return (
        f"❌ I could not find a specific place named **{safe_name}** in the available database. "
        "As an alternative, here are places with a similar type, style, or thematic affinity."
    )


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
    "fala", "para", "from", "with", "there", "happening", "temos", "tem",
    "this", "week", "today", "tomorrow", "next", "year",
    "ano", "esta", "semana", "este", "proxima", "proximo",
}
_EVENT_SPECIFIC_LOOKUP_HINT_TOKENS = {
    "book", "fair", "feira", "fado", "concert", "concerto", "festival", "exhibition",
    "exposicao", "exposição", "music", "musica", "música", "theatre", "teatro",
    "summit", "conference", "congress", "forum", "expo",
}
_EVENT_CATEGORY_FILTER_TERMS = {
    "ao vivo", "live music", "musica", "música", "music", "concert", "concerts",
    "concerto", "concertos", "fado", "jazz", "festival", "festivals", "teatro",
    "theatre", "theater", "dance", "dança", "danca", "cinema", "exhibition",
    "exhibitions", "exposicao", "exposição", "exposicoes", "exposições",
    "gastronomy", "gastronomia", "food", "free", "gratuito", "gratuitos",
    "gratuita", "gratuitas", "familia", "família", "children", "kids",
    "criancas", "crianças", "outdoor", "outdoors", "ar livre",
}
_EVENT_NAMED_ENTITY_HINTS = {
    "book fair", "feira do livro", "web summit", "rock in rio", "doclisboa",
    "lisboa games week", "open house", "lisbon week",
}


def _extract_specific_event_lookup_phrase(query: Optional[str]) -> Optional[str]:
    """Extracts a specific event name from a natural-language query when present."""
    prefixed = _extract_prefixed_lookup_phrase(
        query,
        noise_tokens=_EVENT_SPECIFIC_LOOKUP_NOISE_TOKENS,
        max_tokens=8,
    )
    if prefixed:
        return prefixed

    extracted = _extract_named_lookup_phrase(query, _EVENT_SPECIFIC_LOOKUP_NOISE_TOKENS)
    if extracted:
        return extracted

    raw_query = (query or '').strip()
    meaningful_tokens = [
        token for token in _extract_lookup_tokens(raw_query)
        if token not in _EVENT_SPECIFIC_LOOKUP_NOISE_TOKENS
    ]
    has_year_marker = bool(re.search(r"(?:'\d{2}\b|\b(?:19|20)\d{2}\b)", raw_query))

    has_event_hint = any(token in _EVENT_SPECIFIC_LOOKUP_HINT_TOKENS for token in meaningful_tokens)
    if meaningful_tokens and len(meaningful_tokens) <= 6 and (has_year_marker or has_event_hint):
        return " ".join(meaningful_tokens)

    return None


def _is_event_category_filter_query(
    query: Optional[str],
    category: Optional[str],
    date_filter: Optional[str],
    specific_lookup_phrase: Optional[str],
    specific_lookup_requested: bool = False,
) -> bool:
    """Return whether an event query is a thematic/category search, not a named lookup."""
    normalized_query = _normalize_lookup_text(query)
    if not normalized_query:
        return False

    if specific_lookup_requested:
        return False

    if any(named_hint in normalized_query for named_hint in _EVENT_NAMED_ENTITY_HINTS):
        return False

    if _QUOTED_LOOKUP_PATTERN.search(query or ""):
        return False

    query_tokens = set(_extract_lookup_tokens(normalized_query))
    category_tokens = set(_extract_lookup_tokens(category)) if category else set()
    category_filter_hit = bool(category_tokens and query_tokens & category_tokens)
    thematic_hit = any(term in normalized_query for term in _EVENT_CATEGORY_FILTER_TERMS)

    if category and thematic_hit:
        return True
    if date_filter and thematic_hit and specific_lookup_phrase:
        return True
    if category_filter_hit and thematic_hit:
        return True

    return False


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
    'outdoor': ['outdoors', 'open', 'air', 'park', 'parque', 'garden', 'jardim', 'grass', 'picnic'],
    'outdoors': ['outdoor', 'open', 'air', 'park', 'parque', 'garden', 'jardim', 'grass', 'picnic'],
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


# ==========================================================================
# Vector Store Connection (Lazy Loading)
# ==========================================================================

_vector_store = None
_vector_store_lock = threading.Lock()


def _get_vector_store():
    """
    Lazily initializes the vector store connection.

    Returns:
        KnowledgeBase: The vector store instance, or None if unavailable.
    """
    global _vector_store

    if _vector_store is False:
        return None
    if _vector_store is not None:
        return _vector_store

    with _vector_store_lock:
        if _vector_store is False:
            return None
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
    "cultural", "culture", "cultura", "stop", "paragem", "gallery", "galleries",
    "galeria", "galerias", "viewpoint", "viewpoints", "near", "area", "zona",
    "relaxed", "calm", "evening", "noite", "saldanha", "transport", "public", "real",
}
_SPECIFIC_PLACE_LOOKUP_TYPE_TOKENS = {
    "museum", "museums", "museu", "museus",
    "monument", "monuments", "monumento", "monumentos",
}
_SPECIFIC_PLACE_LOOKUP_NOISE_TOKENS = _GENERIC_PLACE_QUERY_TOKENS - _SPECIFIC_PLACE_LOOKUP_TYPE_TOKENS
_BROAD_PLACE_LOOKUP_CONNECTORS = {
    "in", "near", "with", "around", "for", "by", "at",
    "em", "perto", "com", "para", "por", "junto",
}
_PLACE_LOOKUP_COMMAND_TOKENS = {
    "show", "me", "find", "give", "list", "recommend", "suggest",
    "mostra", "mostre", "encontra", "lista", "recomenda", "sugere",
    "one", "two", "three", "four", "five", "um", "uma", "dois", "duas", "tres", "três", "quatro", "cinco",
}
_KNOWN_PLACE_LOCATION_HINTS = {
    "belem", "alfama", "chiado", "baixa", "rossio", "oriente", "expo",
    "ajuda", "alcantara", "estrela", "graca", "mouraria", "restelo",
    "beato", "cascais", "sintra", "campo", "sodre", "principe",
}
_EXPLICIT_MUSEUM_MARKERS = {
    "museum", "museu", "maat", "mude", "gulbenkian", "berardo", "mac/ccb", "macccb",
}
_NON_MUSEUM_MONUMENT_MARKERS = {
    "monument", "monastery", "castle", "palace", "church", "tower", "cemetery", "aqueduct",
    "planetarium", "planetario", "planetário",
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


_PROMINENT_MUSEUM_MARKER_WEIGHTS: Tuple[Tuple[str, float], ...] = (
    ("gulbenkian", 0.52),
    ("maat", 0.42),
    ("azulejo", 0.38),
    ("arte antiga", 0.36),
    ("ancient art", 0.36),
    ("national coach", 0.30),
    ("coches", 0.30),
    ("maritime", 0.22),
)


def _coerce_ranking_float(value: Any, default: float = 0.0) -> float:
    """Convert scraped ranking fields to floats without raising on sparse data."""
    try:
        if value is None or value == "":
            return default
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def _score_ranked_place_result(result: Dict[str, Any], query_intent: Optional[str]) -> float:
    """Score broad recommendation results with data-backed popularity signals."""
    full_data = _get_place_by_url(result.get("url", "")) if result.get("url") else None
    title = result.get("title", "")
    category = result.get("category", "")
    description = result.get("short_description", "")
    haystack = _normalize_place_hint_text(
        _join_user_facing_parts(
            [
                title,
                category,
                result.get("url", ""),
                description,
                (full_data or {}).get("short_description", ""),
                (full_data or {}).get("full_description", ""),
            ]
        )
    )

    score = float(result.get("ranking_score") or 0.0)
    if query_intent == "museum_only":
        score *= 0.35
    tripadvisor = (full_data or {}).get("tripadvisor") or {}
    rating = _coerce_ranking_float(result.get("rating"), default=0.0) or _coerce_ranking_float(
        tripadvisor.get("rating"), default=0.0
    )
    reviews = _coerce_ranking_float(result.get("reviews"), default=0.0) or _coerce_ranking_float(
        tripadvisor.get("reviews_count"), default=0.0
    )
    if rating:
        score += min(0.45, (rating / 5.0) * 0.45)
    if reviews:
        score += min(0.42, math.log10(reviews + 1) / 4.5)

    if query_intent == "museum_only" and _is_explicit_museum_candidate(title, result.get("url", ""), haystack):
        score += 0.28
        for marker, weight in _PROMINENT_MUSEUM_MARKER_WEIGHTS:
            if marker in haystack:
                score += weight
                break

    if _is_service_like_place_category(category):
        score -= 0.60
    return score


def _is_explicit_museum_candidate(title: str, url: str = "", extra_text: str = "") -> bool:
    """Returns whether a candidate is explicitly museum-like rather than just monument-like."""
    extra_text_clean = re.sub(r"category\s*:\s*[^\n]+", " ", extra_text or "", flags=re.IGNORECASE)
    extra_text_clean = re.sub(r"museums?\s*&\s*monuments?", " ", extra_text_clean, flags=re.IGNORECASE)
    haystack = _normalize_place_hint_text(" ".join([title or "", url or "", extra_text_clean]))
    title_url_haystack = _normalize_place_hint_text(" ".join([title or "", url or ""]))
    compact_haystack = haystack.replace(" ", "")

    if any(marker in title_url_haystack for marker in _NON_MUSEUM_MONUMENT_MARKERS) and not any(
        marker in title_url_haystack for marker in _EXPLICIT_MUSEUM_MARKERS
    ):
        return False
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


def _is_broad_specific_place_phrase(phrase: Optional[str]) -> bool:
    """Detect broad type/location fragments that should not be treated as named places."""
    tokens = [
        token for token in _extract_lookup_tokens(phrase)
        if token not in _PLACE_LOOKUP_COMMAND_TOKENS and not token.isdigit()
    ]
    if len(tokens) < 2:
        return False
    return (
        tokens[0] in _SPECIFIC_PLACE_LOOKUP_TYPE_TOKENS
        and any(token in _BROAD_PLACE_LOOKUP_CONNECTORS for token in tokens[1:])
    )


def _extract_specific_place_lookup_phrase(query: Optional[str]) -> Optional[str]:
    """Extracts a specific place name from quoted or 'tell me about' queries."""
    prefixed = _extract_prefixed_lookup_phrase(
        query,
        noise_tokens=_SPECIFIC_PLACE_LOOKUP_NOISE_TOKENS,
        max_tokens=6,
        strip_trailing_status_terms=True,
    )
    if prefixed:
        prefixed_tokens = _extract_lookup_tokens(prefixed)
        if (
            len(prefixed_tokens) == 1 and prefixed_tokens[0] in _SPECIFIC_PLACE_LOOKUP_TYPE_TOKENS
        ) or _is_broad_specific_place_phrase(prefixed):
            return None
        return prefixed

    extracted = _extract_named_lookup_phrase(query, _SPECIFIC_PLACE_LOOKUP_NOISE_TOKENS)
    if extracted:
        extracted_tokens = _extract_lookup_tokens(extracted)
        if (
            len(extracted_tokens) == 1 and extracted_tokens[0] in _SPECIFIC_PLACE_LOOKUP_TYPE_TOKENS
        ) or _is_broad_specific_place_phrase(extracted):
            return None
        return extracted

    raw_query = (query or '').strip()
    meaningful_tokens = [
        token for token in _extract_lookup_tokens(raw_query)
        if token not in _SPECIFIC_PLACE_LOOKUP_NOISE_TOKENS
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
        candidate_phrase = " ".join(meaningful_tokens)
        if (
            len(meaningful_tokens) == 1 and meaningful_tokens[0] in _SPECIFIC_PLACE_LOOKUP_TYPE_TOKENS
        ) or _is_broad_specific_place_phrase(candidate_phrase):
            return None
        return candidate_phrase
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

    return re.sub(r"\s+", " ", " ".join(cleaned_parts)).strip(" -")


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


def _query_requests_free_events(query: Optional[str]) -> bool:
    """Returns whether the query explicitly asks for free events."""
    return _text_contains_fuzzy_term(
        query,
        [
            "free",
            "free entry",
            "free admission",
            "gratuito",
            "gratuitos",
            "gratuita",
            "gratuitas",
            "gratis",
            "grátis",
            "entrada gratuita",
        ],
    )


def _event_matches_free_filter(event: Dict[str, Any]) -> bool:
    """Returns True when an event clearly advertises free admission."""
    price_text = _normalize_lookup_text(event.get("price") or "")
    if not price_text:
        return False
    return any(
        token in price_text
        for token in [
            "free",
            "free entry",
            "free admission",
            "gratuito",
            "gratuita",
            "gratis",
            "grátis",
            "entrada gratuita",
        ]
    )


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
    language: str = "en",
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
                            previews.append(_localize_visitlisboa_schedule_text(f"{schedule_name}: {day_label} {hours}", language))
                        else:
                            previews.append(_localize_visitlisboa_schedule_text(f"{day_label}: {hours}", language))
        if previews:
            return previews[:2]

    for schedule in schedules:
        if schedule.get('today'):
            previews.append(_localize_visitlisboa_schedule_text(schedule['today'], language))
            break

    if previews:
        return previews[:1]

    # Fallback: show first two hours entries when no today shortcut is available.
    for schedule in schedules:
        for day_label, hours in list((schedule.get('hours') or {}).items())[:2]:
            previews.append(_localize_visitlisboa_schedule_text(f"{day_label}: {hours}", language))
        if previews:
            break

    if previews:
        return previews[:2]

    for schedule in schedules:
        if schedule.get('summary'):
            previews.append(_localize_visitlisboa_schedule_text(schedule['summary'], language))
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


def _event_within_requested_geography(event: Dict[str, Any], query: Optional[str]) -> bool:
    """Keeps Lisbon event discovery focused on Lisbon-city entries unless AML scope was explicit."""
    location_text = " ".join(
        part for part in [
            str(event.get("location") or "").strip(),
            str(event.get("address") or "").strip(),
        ]
        if part
    )
    return _place_within_requested_geography(location_text, query)


_OUTDOOR_EVENT_QUERY_TERMS = [
    "outdoor", "outdoors", "open air", "open-air", "outside", "ao ar livre",
    "ar livre", "exterior", "outdoor event", "outdoor events",
]

_OUTDOOR_EVENT_EVIDENCE_TERMS = [
    "outdoor", "outdoors", "open air", "open-air", "ao ar livre", "ar livre",
    "park", "parks", "parque", "parques", "garden", "gardens", "jardim", "jardins",
    "grass", "picnic", "beach", "praia", "waterfront", "riverside", "riverfront",
    "miradouro", "viewpoint", "belvedere", "terrace", "terraço", "terraco",
]


def _query_requests_outdoor_events(query: Optional[str]) -> bool:
    """Returns whether the user explicitly asks for outdoor events."""
    normalized_query = _normalize_lookup_text(query)
    if not normalized_query:
        return False
    return any(term in normalized_query for term in _OUTDOOR_EVENT_QUERY_TERMS)


def _event_has_outdoor_context(event: Dict[str, Any]) -> bool:
    """Returns whether an event has textual evidence that it is outdoors."""
    searchable_text = _normalize_lookup_text(_build_event_searchable_text(event))
    return any(term in searchable_text for term in _OUTDOOR_EVENT_EVIDENCE_TERMS)


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


def _place_result_full_data(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return full VisitLisboa place data for a normalized result when available."""
    if result.get("source") != "visitlisboa" or not result.get("url"):
        return {}
    return _get_place_by_url(str(result.get("url") or "")) or {}


def _place_result_has_ticket_link(result: Dict[str, Any]) -> bool:
    """Return whether a place result has a real ticket URL."""
    full_data = _place_result_full_data(result)
    tickets_offers = full_data.get("tickets_offers") or {}
    if isinstance(tickets_offers, dict):
        for label, url in _iter_named_http_links(tickets_offers.get("links")):
            if _is_ticket_http_link(label, url):
                return True
    tickets_url = (full_data.get("contact_info") or {}).get("tickets_url")
    return _is_ticket_http_link("", tickets_url)


def _place_result_has_schedule(result: Dict[str, Any]) -> bool:
    """Return whether a place result has structured opening-hour data."""
    full_data = _place_result_full_data(result)
    schedules = full_data.get("schedules")
    return isinstance(schedules, list) and bool(schedules)


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


def _infer_specific_place_fallback_category(query: Optional[str], category: Optional[str]) -> Optional[str]:
    """Infer a same-type fallback category when a named place lookup has no exact match."""
    if category:
        return category

    query_intent = _infer_place_query_intent(query, category)
    if query_intent in {"museum_only", "museum_monument", "monument_only"}:
        return "Museums & Monuments"
    if query_intent == "food":
        return "Restaurants"
    if query_intent == "accommodation":
        return "Hotels"
    if _text_contains_fuzzy_term(query, ["viewpoint", "view point", "miradouro", "miradouros"]):
        return "View Points"
    if _text_contains_fuzzy_term(query, ["park", "parks", "garden", "gardens", "parque", "parques", "jardim", "jardins"]):
        return "Parks & Gardens"
    return None


def _infer_specific_event_fallback_category(query: Optional[str], category: Optional[str]) -> Optional[str]:
    """Infer a same-type fallback category when a named event lookup has no exact match."""
    if category:
        return category

    normalized_query = _normalize_lookup_text(query)
    if not normalized_query:
        return None

    category_rules = [
        (["summit", "conference", "congress", "forum", "expo", "technology", "tech", "startup"], "Main Events"),
        (["music", "concert", "concerto", "fado", "jazz", "rock", "pop"], "Music"),
        (["theatre", "theater", "teatro", "opera", "dance", "danca", "dança", "ballet"], "Theater Opera & Dance"),
        (["festival", "festivals", "festivais"], "Festivals"),
        (["exhibition", "exhibitions", "exposicao", "exposição", "art", "arte", "gallery", "galeria"], "Exhibitions"),
        (["sport", "sports", "desporto", "desportos", "marathon", "maratona", "grand prix", "triathlon"], "Sports"),
        (["cinema", "film", "movie", "movies"], "Cinema"),
        (["fair", "fairs", "feira", "feiras", "market", "mercado"], "Fairs"),
        (["food", "gastronomy", "gastronomia", "wine", "vinho", "culinary"], "Gastronomy"),
    ]

    for terms, fallback_category in category_rules:
        if any(term in normalized_query for term in terms):
            return fallback_category
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
        service_anchor_text = _join_user_facing_parts([
            item.get('title', ''),
            item.get('category', ''),
            item.get('location', ''),
            item.get('short_description', ''),
        ])

        geography_text = _join_user_facing_parts([item.get('title', ''), item.get('url', ''), item.get('location', '')])
        if not _place_within_requested_geography(geography_text, query):
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
    specific_lookup: bool = False,
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
        category_filter_query = _is_event_category_filter_query(
            query,
            category,
            date_filter,
            specific_lookup_phrase,
            specific_lookup_requested=specific_lookup,
        )
        if category_filter_query:
            specific_lookup_phrase = None
        if specific_lookup and query and not specific_lookup_phrase:
            specific_lookup_phrase = _normalize_lookup_text(query)
        effective_query = specific_lookup_phrase or query
        free_filter_requested = _query_requests_free_events(effective_query or query)
        outdoor_filter_requested = _query_requests_outdoor_events(effective_query or query)
        if free_filter_requested and effective_query:
            effective_query = re.sub(
                r"\b(?:free(?:\s+entry|\s+admission)?|gratuit[oa]s?|gratis|gr[aá]tis|entrada\s+gratuita)\b",
                " ",
                effective_query,
                flags=re.IGNORECASE,
            )
            effective_query = re.sub(r"\s+", " ", effective_query).strip(" .?!") or None
            if specific_lookup_phrase and not effective_query:
                specific_lookup_phrase = None

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

        if free_filter_requested:
            events_data = [event for event in events_data if _event_matches_free_filter(event)]
            undated_candidates = [event for event in undated_candidates if _event_matches_free_filter(event)]
            logger.info(f"After free-event filter: {len(events_data)} events")

        if outdoor_filter_requested:
            events_data = [event for event in events_data if _event_has_outdoor_context(event)]
            undated_candidates = [event for event in undated_candidates if _event_has_outdoor_context(event)]
            logger.info(f"After outdoor-event filter: {len(events_data)} events")

        category_filtered_pool = list(events_data)

        # Step 3: Filter by query (TOKEN-BASED matching for better recall)
        query_scores: Dict[int, float] = {}
        strong_specific_event_ids: set[int] = set()
        if query:
            expanded_tokens = _expand_event_query_tokens(effective_query)

            if specific_lookup_phrase or expanded_tokens:
                def _score_collection(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[int, float], set[int]]:
                    scores: Dict[int, float] = {}
                    matched_items: List[Dict[str, Any]] = []
                    strong_ids: set[int] = set()
                    for item in items:
                        score = _score_event_query_match(
                            item,
                            expanded_tokens,
                            specific_lookup_phrase=specific_lookup_phrase,
                        )
                        if score > 0:
                            scores[id(item)] = score
                            matched_items.append(item)
                            if specific_lookup_phrase and _is_strong_specific_event_match(
                                item,
                                score,
                                specific_lookup_phrase,
                                expanded_tokens,
                            ):
                                strong_ids.add(id(item))
                    return matched_items, scores, strong_ids

                events_data, query_scores, strong_specific_event_ids = _score_collection(events_data)
                undated_candidates, _, _ = _score_collection(undated_candidates)
                logger.info(
                    f"After query filter: {len(events_data)} events (specific_lookup={bool(specific_lookup_phrase)}, tokens={expanded_tokens[:6]})"
                )
            else:
                logger.info(f"Query '{query}' contained only generic terms, skipping text filter.")

        exact_lookup_not_found_intro: Optional[str] = None
        if not events_data and specific_lookup_phrase:
            fallback_category = _infer_specific_event_fallback_category(effective_query or query, category)
            fallback_candidates = list(category_filtered_pool)
            if fallback_category and not category:
                fallback_category_lower = fallback_category.lower()
                fallback_candidates = [
                    event for event in fallback_candidates
                    if fallback_category_lower in event.get('category', '').lower()
                ]

            if fallback_candidates:
                events_data = fallback_candidates
                query_scores = {id(event): 0.0 for event in events_data}
                exact_lookup_not_found_intro = _build_specific_lookup_fallback_intro(
                    specific_lookup_phrase,
                    language=render_language,
                    content_kind="event",
                )

        if not events_data:
            localized_date_filter = _localize_event_date_filter(date_filter, language=render_language)
            localized_category = _localize_event_category(category, language=render_language) if category else None
            if specific_lookup_phrase:
                return _build_specific_lookup_fallback_intro(
                    specific_lookup_phrase,
                    language=render_language,
                    content_kind="event",
                )
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
            event['_relevance_score'] = calculate_temporal_relevance_score(event, start_date)
            event['_duration_days'] = get_event_duration_days(event)
            event['_query_match_score'] = query_scores.get(id(event), 0.0)
            event['_geography_score'] = 1 if _event_within_requested_geography(event, query) else 0

        if specific_lookup_phrase:
            exact_matches = [event for event in events_data if id(event) in strong_specific_event_ids]
            if exact_matches:
                events_data = exact_matches
                exact_lookup_not_found_intro = None
            elif exact_lookup_not_found_intro is None:
                exact_lookup_not_found_intro = _build_specific_lookup_fallback_intro(
                    specific_lookup_phrase,
                    language=render_language,
                    content_kind="event",
                )

        if query_scores:
            events_data.sort(
                key=lambda e: (
                    e.get('_geography_score', 0),
                    e.get('_query_match_score', 0.0),
                    e.get('_relevance_score', 0.0),
                ),
                reverse=True,
            )
            logger.info(
                "Sorted by geography, query relevance, and temporal score (top geo: %d, query: %.1f, temporal: %.1f)",
                events_data[0].get('_geography_score', 0),
                events_data[0].get('_query_match_score', 0.0),
                events_data[0].get('_relevance_score', 0.0),
            )
        else:
            events_data.sort(
                key=lambda e: (e.get('_geography_score', 0), e.get('_relevance_score', 0)),
                reverse=True,
            )
            logger.info(
                "Sorted by geography and temporal relevance (top geo: %d, top temporal score: %.1f)",
                events_data[0].get('_geography_score', 0),
                events_data[0].get('_relevance_score', 0.0),
            )

        if offset >= len(events_data):
            output_parts = _format_event_filter_summary(
                query=query,
                category=category,
                date_filter=date_filter,
                start_date=start_date,
                end_date=end_date,
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

        # Keep final cards concise and aligned with the response-quality contract.
        display_cap = min(max_results, MAX_USER_FACING_RESULTS)
        display_count = min(display_cap, 2) if exact_lookup_not_found_intro and offset == 0 else display_cap

        # Limit results
        results = events_data[offset : offset + display_count]

        # Format output with contextual filter summary and concise descriptions
        output_parts = _format_event_filter_summary(
            query=query,
            category=category,
            date_filter=date_filter,
            start_date=start_date,
            end_date=end_date,
            language=render_language,
        )
        if exact_lookup_not_found_intro and offset == 0:
            output_parts = [exact_lookup_not_found_intro, "", *output_parts]
        output_parts.append("")

        for _i, event in enumerate(results, 1):
            title = _localize_event_title(
                _clean_event_title(event.get('title'), event.get('url', '')),
                language=render_language,
            )
            cat = _localize_event_category(event.get('category', 'General'), language=render_language)
            loc = event.get('location', 'Lisbon')
            venue_name = str(event.get('venue_name') or '').strip()
            if venue_name and venue_name.lower() not in loc.lower():
                loc = f"{venue_name}, {loc}"
            dates_str = format_event_dates(event, language=render_language)
            duration = event.get('_duration_days', get_event_duration_days(event))
            duration_label = _format_event_duration_label(duration, language=render_language)
            description_summary = _localize_visitlisboa_description(
                _summarize_event_description(event.get('short_description') or event.get('full_description')),
                language=render_language,
            )
            if "descrição disponível na página oficial" in description_summary.lower():
                description_summary = ""
            price_text = _localize_event_price(event.get('price'), language=render_language)

            event_icon = _event_icon_for_category(cat)
            output_parts.append(f"**{event_icon} {title}**")
            if render_language == "pt":
                output_parts.append(f"    - 🗓️ **Quando:** {dates_str}")
                output_parts.append(f"    - ⏱️ **Duração:** {duration_label}")
                output_parts.append(f"    - 📂 **Categoria:** {cat}")
            else:
                output_parts.append(f"    - 🗓️ **When:** {dates_str}")
                output_parts.append(f"    - ⏱️ **Duration:** {duration_label}")
                output_parts.append(f"    - 📂 **Category:** {cat}")

            if description_summary:
                if render_language == "pt":
                    output_parts.append(f"    - 📝 **Descrição:** {description_summary}")
                else:
                    output_parts.append(f"    - 📝 **Description:** {description_summary}")

            location_lines: List[str] = []
            venue_locations = event.get("venue_locations")
            if isinstance(venue_locations, list) and venue_locations:
                for venue in venue_locations[:2]:
                    if not isinstance(venue, dict):
                        continue
                    venue_label = str(venue.get("venue_name") or "").strip()
                    venue_location = str(venue.get("location") or "").strip()
                    if venue_label and venue_location and venue_label.lower() not in venue_location.lower():
                        location_value = f"{venue_label}, {venue_location}"
                    else:
                        location_value = venue_location or venue_label
                    line = _format_visitlisboa_location_line(
                        location_value,
                        title,
                        language=render_language,
                        label_override="Local" if render_language == "pt" else "Venue",
                    )
                    if line and line not in location_lines:
                        location_lines.append(line)
            else:
                location_line = _format_visitlisboa_location_line(
                    loc,
                    title,
                    language=render_language,
                    label_override="Local" if render_language == "pt" else "Venue",
                )
                if location_line:
                    location_lines.append(location_line)

            output_parts.extend(location_lines)

            # Show price information if available
            if price_text:
                if render_language == "pt":
                    output_parts.append(f"    - 💶 **Preço:** {price_text}")
                else:
                    output_parts.append(f"    - 💶 **Price:** {price_text}")

            official_link = _first_named_http_link(event.get("information_links"))
            if official_link:
                _, official_url = official_link
                official_url = _preserve_source_url(official_url)
                link_text = "Website oficial" if render_language == "pt" else "Official website"
                label = "Website"
                output_parts.append(f"    - 🌐 **{label}:** {_format_markdown_link(link_text, official_url)}")

            # Show buy tickets link if available
            if event.get('buy_tickets_url') and _is_http_url(event.get('buy_tickets_url')):
                if render_language == "pt":
                    output_parts.append(
                        f"    - 🎟️ **Bilhetes:** {_format_markdown_link('Comprar bilhetes', event['buy_tickets_url'])}"
                    )
                else:
                    output_parts.append(
                        f"    - 🎟️ **Tickets:** {_format_markdown_link('Buy tickets', event['buy_tickets_url'])}"
                    )

            if event.get('schedule_notes'):
                schedule_summary = "; ".join(
                    _localize_visitlisboa_schedule_text(note, language=render_language)
                    for note in event['schedule_notes'][:2]
                    if _clean_user_facing_value(note)
                )
                if schedule_summary:
                    if render_language == "pt":
                        output_parts.append(f"    - 🕒 **Horários:** {schedule_summary}")
                    else:
                        output_parts.append(f"    - 🕒 **Schedule:** {schedule_summary}")

            if event.get('highlight_links'):
                highlight_titles = " · ".join(
                    _format_markdown_link(
                        str(item.get('title') or "Highlight"),
                        _preserve_source_url(item.get("url")),
                    )
                    or _clean_user_facing_value(item.get("title"))
                    for item in event['highlight_links'][:3]
                    if isinstance(item, dict) and item.get('title')
                )
                if highlight_titles:
                    if render_language == "pt":
                        output_parts.append(f"    - ✨ **Destaques:** {highlight_titles}")
                    else:
                        output_parts.append(f"    - ✨ **Highlights:** {highlight_titles}")

            if event.get('url'):
                details_label = "Mais detalhes" if render_language == "pt" else "More details"
                details_url = _preserve_source_url(event['url'])
                output_parts.append(f"    - 🔗 **{details_label}:** {_format_markdown_link('VisitLisboa', details_url)}")

            output_parts.append("")  # Empty line between events

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
    specific_lookup: bool = False,
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
        if specific_lookup and query and not specific_lookup_query:
            specific_lookup_query = _normalize_lookup_text(query)
        effective_query = _apply_known_place_lookup_alias(specific_lookup_query or query)
        if specific_lookup_query:
            specific_lookup_query = _apply_known_place_lookup_alias(specific_lookup_query) or specific_lookup_query
        query_intent = _infer_place_query_intent(effective_query or query, category)
        query_context = query or effective_query or ""
        tickets_requested = _query_mentions_tickets(query_context)
        schedule_requested = _query_mentions_schedule(query_context)

        logger.info(
            f"search_places_attractions: query='{query}', effective_query='{effective_query}', category='{category}', max={max_results}, offset={offset}"
        )
        required_service_terms = _extract_required_service_term_groups(effective_query or query)

        # Check if we should also search Dados Abertos (hybrid mode)
        search_dados_abertos = _should_search_dados_abertos(effective_query or query)
        requested_category_for_hybrid = _normalize_place_category_filter(category)
        if requested_category_for_hybrid in {"museums & monuments", "restaurants", "hotels", "view points", "tours"}:
            search_dados_abertos = False
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

                query_context = query or effective_query or search_query

                query_tokens = _extract_place_query_tokens(effective_query or search_query)
                location_hints = _extract_place_location_hints(effective_query or search_query)
                ranking_requested = _query_requests_ranked_places(effective_query or search_query) or query_intent == "top_attractions"
                lisboa_card_requested = _query_mentions_lisboa_card(query_context)
                tickets_requested = _query_mentions_tickets(query_context)
                schedule_requested = _query_mentions_schedule(query_context)

                logger.info(f"search_places_attractions: searching VisitLisboa for '{search_query}'")

                search_k = requested_window * 2
                if query_intent in {"top_attractions", "museum_monument"} and not specific_lookup_query:
                    search_k = max(search_k, requested_window * 5, 15)

                results_with_scores = kb.search_with_scores(
                    query=search_query,
                    k=search_k,
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
                    service_anchor_text = _join_user_facing_parts([
                        metadata.get('title', ''),
                        item_category,
                        metadata.get('url', ''),
                        raw_location,
                        cleaned_doc_description,
                    ])

                    geography_text = _join_user_facing_parts([metadata.get('title', ''), metadata.get('url', ''), raw_location])
                    if not _place_within_requested_geography(geography_text, query):
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
                selection_window = requested_window
                if tickets_requested or schedule_requested:
                    selection_window = max(requested_window * 4, 12)
                for item in scored_candidates[:selection_window]:
                    metadata = item['metadata']
                    # Attempt to get real address/location
                    full_place_data = _get_place_by_url(metadata.get('url', '')) if metadata.get('url') else None
                    real_location = (
                        (full_place_data or {}).get('address')
                        or (full_place_data or {}).get('location')
                        or metadata.get('address')
                        or metadata.get('location')
                        or 'Lisbon'
                    )
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

                has_catalog_backing = any(
                    bool(_get_place_by_url(str(result.get("url") or "")))
                    for result in visitlisboa_results
                    if result.get("url")
                )
                if ranking_requested and query_intent == "museum_only" and not location_hints and has_catalog_backing:
                    json_candidates = _fallback_search(
                        query=None,
                        category=category or "Museums & Monuments",
                        data=_load_places_json(),
                        max_results=5000,
                    )
                    json_candidates = [
                        item
                        for item in json_candidates
                        if _is_explicit_museum_candidate(
                            item.get("title", ""),
                            item.get("url", ""),
                            _build_place_searchable_text(item),
                        )
                    ]
                    json_results = [_convert_raw_place_to_result(item) for item in json_candidates]
                    merged_ranked_results: List[Dict[str, Any]] = []
                    seen_ranked_keys: set[str] = set()
                    _append_unique_place_results(merged_ranked_results, visitlisboa_results, seen_ranked_keys)
                    _append_unique_place_results(merged_ranked_results, json_results, seen_ranked_keys)
                    visitlisboa_results = sorted(
                        merged_ranked_results,
                        key=lambda result: _score_ranked_place_result(result, query_intent),
                        reverse=True,
                    )[:requested_window]

            except Exception as e:
                logger.warning(f"Vector search failed: {e}")

        # JSON fallback to improve recall when vector search under-recovers
        should_use_json_fallback = bool(effective_query or query) and (
            not visitlisboa_results
            or bool(specific_lookup_query)
            or query_intent != "top_attractions"
        )
        if should_use_json_fallback and (len(visitlisboa_results) < requested_window or tickets_requested or schedule_requested):
            fallback_items = _fallback_search(
                query=effective_query or query,
                category=category,
                data=_load_places_json(),
                max_results=5000 if (tickets_requested or schedule_requested) else requested_window * 2,
            )
            fallback_results = [_convert_raw_place_to_result(item) for item in fallback_items]
            if tickets_requested or schedule_requested:
                supplemental_items = _fallback_search(
                    query=None,
                    category=category,
                    data=_load_places_json(),
                    max_results=5000,
                )
                fallback_results.extend(_convert_raw_place_to_result(item) for item in supplemental_items)
            combined_visitlisboa: List[Dict[str, Any]] = []
            seen_visitlisboa_keys: set[str] = set()
            _append_unique_place_results(combined_visitlisboa, visitlisboa_results, seen_visitlisboa_keys)
            fallback_limit = None if (tickets_requested or schedule_requested) else requested_window
            _append_unique_place_results(combined_visitlisboa, fallback_results, seen_visitlisboa_keys, limit=fallback_limit)
            visitlisboa_results = combined_visitlisboa

        if required_service_terms:
            visitlisboa_results = [
                result
                for result in visitlisboa_results
                if _matches_required_service_terms(
                    _join_user_facing_parts(
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

        if effective_query or query:
            output_location_hints = _extract_place_location_hints(effective_query or query)
            if output_location_hints:
                all_results.sort(
                    key=lambda result: (
                        0 if _matches_place_location_hints(
                            _join_user_facing_parts(
                                [
                                    result.get("title", ""),
                                    result.get("location", ""),
                                    result.get("short_description", ""),
                                    result.get("url", ""),
                                ]
                            ),
                            output_location_hints,
                        ) else 1,
                        -float(result.get("ranking_score") or 0.0),
                    )
                )

        if tickets_requested or schedule_requested:
            all_results.sort(
                key=lambda result: (
                    1 if (not tickets_requested or _place_result_has_ticket_link(result)) else 0,
                    1 if (not schedule_requested or _place_result_has_schedule(result)) else 0,
                    float(result.get("ranking_score") or 0.0),
                ),
                reverse=True,
            )

        # =====================================================================
        # STEP 3: Format Output
        # =====================================================================

        exact_lookup_not_found_intro: Optional[str] = None

        if specific_lookup_query and all_results:
            exact_matches: List[Dict[str, Any]] = []
            perfect_signature_matches: List[Dict[str, Any]] = []
            requested_signature = _normalize_specific_place_signature(specific_lookup_query)
            for result in all_results:
                candidate = result
                if result.get('url') and result.get('source') == 'visitlisboa':
                    full_place = _get_place_by_url(result['url'])
                    if full_place:
                        candidate = full_place
                score = _score_specific_place_lookup_match(candidate, specific_lookup_query)
                if _is_strong_specific_place_match(score, specific_lookup_query):
                    exact_matches.append(result)
                    if requested_signature and _normalize_specific_place_signature(candidate.get('title')) == requested_signature:
                        perfect_signature_matches.append(result)

            if perfect_signature_matches:
                all_results = perfect_signature_matches
            elif exact_matches:
                all_results = exact_matches
            else:
                exact_lookup_not_found_intro = _build_specific_lookup_fallback_intro(
                    specific_lookup_query,
                    language=render_language,
                    content_kind="place",
                )

        if not all_results:
            # Last resort fallback
            if effective_query or query:
                fallback_items = _fallback_search(effective_query or query, category, _load_places_json(), max_results=requested_window)
                if fallback_items:
                    all_results = [_convert_raw_place_to_result(item) for item in fallback_items]
                    if specific_lookup_query and not exact_lookup_not_found_intro:
                        exact_lookup_not_found_intro = _build_specific_lookup_fallback_intro(
                            specific_lookup_query,
                            language=render_language,
                            content_kind="place",
                        )

            if not all_results and specific_lookup_query:
                fallback_category = _infer_specific_place_fallback_category(specific_lookup_query, category)
                if fallback_category:
                    fallback_items = _fallback_search(None, fallback_category, _load_places_json(), max_results=requested_window)
                    if fallback_items:
                        all_results = [_convert_raw_place_to_result(item) for item in fallback_items]
                        if not exact_lookup_not_found_intro:
                            exact_lookup_not_found_intro = _build_specific_lookup_fallback_intro(
                                specific_lookup_query,
                                language=render_language,
                                content_kind="place",
                            )

            if not all_results and (effective_query or query):
                logger.info("No results from hybrid search, trying direct Dados Abertos")
                from tools.dados_abertos import _search_place_in_datasets_logic
                open_data_results = _search_place_in_datasets_logic(effective_query or query, max_results=requested_window)
                if open_data_results:
                    if specific_lookup_query:
                        normalized_specific_lookup = _normalize_lookup_text(specific_lookup_query)
                        normalized_open_data_output = _normalize_lookup_text(open_data_results)
                        if normalized_specific_lookup and normalized_specific_lookup not in normalized_open_data_output:
                            intro = _build_specific_lookup_fallback_intro(
                                specific_lookup_query,
                                language=render_language,
                                content_kind="place",
                            )
                            return f"{intro}\n\n{open_data_results}".strip()
                    return open_data_results

            if specific_lookup_query and not all_results:
                return _build_specific_lookup_fallback_intro(
                    specific_lookup_query,
                    language=render_language,
                    content_kind="place",
                )

            if not all_results and render_language == "pt":
                return f"Não foram encontrados locais correspondentes a '{query or 'todos'}' no VisitLisboa ou nos registos de dados abertos."
            if not all_results:
                return f"No places found matching: '{query or 'all'}' in VisitLisboa or Open Data registries."

        # Keep cards concise and aligned with the response-quality contract.
        display_cap = min(max_results, MAX_USER_FACING_RESULTS)
        display_count = min(display_cap, 2) if exact_lookup_not_found_intro and offset == 0 else display_cap

        # Limit to the requested window.
        final_results = all_results[offset : offset + display_count]
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

        if render_language == "pt":
            output_parts = [
                "### 🔵 **Locais e atrações**",
                f"🧭 **Janela de resultados:** {offset + 1}-{offset + len(final_results)} de {len(all_results)}.",
            ]
        else:
            output_parts = [
                "### 🔵 **Places and Attractions**",
                f"🧭 **Result window:** {offset + 1}-{offset + len(final_results)} of {len(all_results)}.",
            ]

        if exact_lookup_not_found_intro and offset == 0:
            output_parts = [exact_lookup_not_found_intro, "", *output_parts]

        for _i, place in enumerate(final_results, 1):
            title = _localize_place_title(place.get('title', 'Unknown'), language=render_language)
            cat = _localize_place_category(place.get('category', 'General'), language=render_language)
            source = place.get('source', 'unknown')

            # Try to get full data from JSON for richer output
            full_data = None
            if place.get('url') and source == 'visitlisboa':
                full_data = _get_place_by_url(place['url'])
            loc = (
                (full_data or {}).get('address')
                or (full_data or {}).get('location')
                or place.get('address')
                or place.get('location')
                or 'Lisbon'
            )

            # Source indicator
            place_icon = _place_icon_for_category(place.get('category', cat))
            output_parts.append("")
            output_parts.append(f"**{place_icon} {title}**")

            description_text = _clean_place_description_text(
                (full_data.get('short_description') if full_data else None) or place.get('short_description'),
                title,
            )
            description_text = _localize_visitlisboa_description(
                description_text,
                language=render_language,
            )
            known_description = _known_place_description(title, language=render_language)
            generic_description = _normalize_lookup_text(description_text) in {
                "descricao disponivel na pagina oficial do local",
                "description available on the official place page",
            }
            if known_description and (not description_text or generic_description):
                description_text = known_description
            elif generic_description:
                description_text = ""
            if description_text:
                desc = description_text[:220]
                if len(description_text) > 220:
                    desc = desc.rsplit(" ", 1)[0].rstrip(" ,;:.") + "…"
                description_label = "Descrição" if render_language == "pt" else "Description"
                output_parts.append(f"    - 📝 **{description_label}:** {desc}")

            output_parts.append(f"    - 📂 **{'Categoria' if render_language == 'pt' else 'Category'}:** {cat}")

            location_line = _format_visitlisboa_location_line(loc, title, language=render_language)
            if location_line:
                output_parts.append(location_line)
            elif place.get('lat') and place.get('lon'):
                map_label = "Mapa" if render_language == "pt" else "Map"
                address_label = "Localização" if render_language == "pt" else "Location"
                coordinate_line = _format_coordinates_location_line(place.get('lat'), place.get('lon'), address_label, map_label)
                if coordinate_line:
                    output_parts.append(coordinate_line)

            feature_summary = _format_compact_feature_summary(
                full_data.get("features") if full_data else None,
                language=render_language,
            )
            if feature_summary:
                label = "Características" if render_language == "pt" else "Features"
                output_parts.append(f"    - ✨ **{label}:** {feature_summary}")

            # Schedule/opening hours (from enriched data)
            if full_data and full_data.get('schedules'):
                for preview in _format_place_schedule_previews(
                    full_data['schedules'],
                    query or effective_query,
                    language=render_language,
                ):
                    label = "Horário" if render_language == "pt" else "Hours"
                    output_parts.append(f"    - 🕒 **{label}:** {preview}")

            # Tickets/prices (from enriched data)
            price_values: List[str] = []
            ticket_line_added = False
            lisboa_card_benefit = None
            if full_data:
                lisboa_card_benefit = full_data.get('lisboa_card_benefit') or full_data.get('lisboa_card_discount')
            if lisboa_card_benefit:
                localized_benefit = _localize_place_value_text(lisboa_card_benefit, language=render_language)
                if localized_benefit:
                    price_values.append(localized_benefit)
            if full_data and full_data.get('tickets_offers'):
                tickets = full_data['tickets_offers']
                if tickets.get('description'):
                    price_desc = _compact_place_ticket_price_text(tickets['description'], language=render_language)
                    if price_desc:
                        price_values.append(price_desc)
            if price_values:
                label = "Preço" if render_language == "pt" else "Price"
                output_parts.append(f"    - 💶 **{label}:** {'; '.join(dict.fromkeys(price_values))}")

            if full_data and full_data.get('tickets_offers') and full_data['tickets_offers'].get('links'):
                ticket_link = _first_ticket_http_link(full_data['tickets_offers'].get('links'))
                if ticket_link:
                    _, ticket_url = ticket_link
                    if render_language == "pt":
                        output_parts.append(f"    - 🎟️ **Bilhetes:** {_format_markdown_link('Comprar bilhetes', ticket_url)}")
                    else:
                        output_parts.append(f"    - 🎟️ **Tickets:** {_format_markdown_link('Buy tickets', ticket_url)}")
                    ticket_line_added = True
            elif full_data and full_data.get('contact_info', {}).get('tickets_url') and _query_mentions_tickets(query or effective_query):
                tickets_url = full_data['contact_info']['tickets_url']
                if _is_ticket_http_link("", tickets_url):
                    if render_language == "pt":
                        output_parts.append(f"    - 🎟️ **Bilhetes:** {_format_markdown_link('Comprar bilhetes', tickets_url)}")
                    else:
                        output_parts.append(f"    - 🎟️ **Tickets:** {_format_markdown_link('Buy tickets', tickets_url)}")
                    ticket_line_added = True

            # TripAdvisor rating (from enriched data)
            if full_data and full_data.get('tripadvisor'):
                ta = full_data['tripadvisor']
                if ta.get('rating'):
                    reviews_label = "avaliações" if render_language == "pt" else "reviews"
                    rating_value = _clean_user_facing_value(ta.get('rating'))
                    reviews_count = _clean_user_facing_value(ta.get('reviews_count'))
                    rating_label = "Avaliação" if render_language == "pt" else "Rating"
                    if reviews_count:
                        output_parts.append(f"    - ⭐ **{rating_label}:** TripAdvisor {rating_value}/5 ({reviews_count} {reviews_label})")
                    else:
                        output_parts.append(f"    - ⭐ **{rating_label}:** TripAdvisor {rating_value}/5")

            # Contact info (from enriched data)
            if full_data and full_data.get('contact_info'):
                contact = full_data['contact_info']
                if contact.get('phone'):
                    phone_link = _format_phone_link(contact.get('phone'))
                    if phone_link:
                        label = "Telefone" if render_language == "pt" else "Phone"
                        output_parts.append(f"    - 📞 **{label}:** {phone_link}")
                if contact.get('email'):
                    email = _clean_user_facing_value(contact.get('email'))
                    if email:
                        label = "Email"
                        output_parts.append(f"    - ✉️ **{label}:** [{email}](mailto:{email})")
                if contact.get('website') and _is_http_url(contact.get('website')):
                    label = "Website"
                    link_text = "Website oficial" if render_language == "pt" else "Official website"
                    output_parts.append(f"    - 🌐 **{label}:** {_format_markdown_link(link_text, contact.get('website'))}")

            if not ticket_line_added and full_data and full_data.get('tickets_offers') and full_data['tickets_offers'].get('links'):
                ticket_link = _first_ticket_http_link(full_data['tickets_offers'].get('links'))
                if ticket_link:
                    _, ticket_url = ticket_link
                    if render_language == "pt":
                        output_parts.append(f"    - 🎟️ **Bilhetes:** {_format_markdown_link('Comprar bilhetes', ticket_url)}")
                    else:
                        output_parts.append(f"    - 🎟️ **Tickets:** {_format_markdown_link('Buy tickets', ticket_url)}")
            elif not ticket_line_added and full_data and full_data.get('contact_info', {}).get('tickets_url') and _query_mentions_tickets(query or effective_query):
                tickets_url = full_data['contact_info']['tickets_url']
                if _is_ticket_http_link("", tickets_url):
                    if render_language == "pt":
                        output_parts.append(f"    - 🎟️ **Bilhetes:** {_format_markdown_link('Comprar bilhetes', tickets_url)}")
                    else:
                        output_parts.append(f"    - 🎟️ **Tickets:** {_format_markdown_link('Buy tickets', tickets_url)}")

            if place.get('url'):
                details_label = "Mais detalhes" if render_language == "pt" else "More details"
                details_url = _preserve_source_url(place['url'])
                output_parts.append(f"    - 🔗 **{details_label}:** {_format_markdown_link('VisitLisboa', details_url)}")

        # Source breakdown
        return "\n".join(output_parts)

    except Exception as e:
        logger.error(f"Error in search_places_attractions: {e}")
        return f"❌ Error searching places: {str(e)}"


@tool
def get_event_categories(language: str = "en") -> str:
    """
    Get all available event categories from VisitLisboa data.

    Args:
        language: Output language, either ``"en"`` or ``"pt"``.

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

        is_pt = str(language or "en").lower().startswith("pt")
        event_word = "evento" if is_pt else "event"
        event_word_plural = "eventos" if is_pt else "events"
        title = "Categorias de Eventos em Lisboa" if is_pt else "Event Categories in Lisbon"
        emoji_map = {
            "music": "🎵",
            "exhibitions": "🖼️",
            "main events": "🎟️",
            "fairs": "🎪",
            "festivals": "🎉",
            "others": "📌",
            "theater opera & dance": "🎭",
            "theatre opera & dance": "🎭",
            "sports": "⚽",
            "cinema": "🎬",
            "uncategorized": "📁",
        }

        output_parts = [f"### 🎭 **{title}**\n"]
        for cat, count in sorted_categories:
            normalized_cat = _normalize_lookup_text(cat)
            emoji = emoji_map.get(normalized_cat, "📌")
            label = _localize_event_category(cat, language="pt") if is_pt else str(cat).strip()
            if not label or normalized_cat == "uncategorized":
                label = "Sem categoria" if is_pt else "Uncategorized"
            noun = event_word if count == 1 else event_word_plural
            output_parts.append(f"- {emoji} **{label}:** {count} {noun}")

        total_label = "Total de eventos" if is_pt else "Total events"
        tip = (
            "💡 **Dica:** Podes perguntar por uma categoria específica para resultados com datas e locais."
            if is_pt
            else "💡 **Tip:** Ask for a specific category to get dated events and venues."
        )
        output_parts.append(f"\n📊 **{total_label}:** {len(events_data)}")
        output_parts.append(f"\n{tip}")

        return "\n".join(output_parts)

    except Exception as e:
        logger.error(f"Error in get_event_categories: {e}")
        return f"❌ Error getting event categories: {str(e)}"


@tool
def get_place_categories(language: str = "en") -> str:
    """
    Get all available place categories from VisitLisboa data.

    Args:
        language: Output language, either ``"en"`` or ``"pt"``.

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

        is_pt = str(language or "en").lower().startswith("pt")
        category_counts = {}
        title = "Categorias de Locais Disponíveis" if is_pt else "Available Place Categories"
        place_word = "local" if is_pt else "place"
        place_word_plural = "locais" if is_pt else "places"
        category_groups = [
            {
                "keys": {"museums", "museums & monuments", "monuments"},
                "emoji": "🏛️",
                "en": "Museums & Monuments",
                "pt": "Museus e Monumentos",
                "en_examples": "Museums, monuments, heritage sites.",
                "pt_examples": "Museus, monumentos e património histórico.",
            },
            {
                "keys": {"tours", "only in lisbon", "attractions"},
                "emoji": "✨",
                "en": "Tours & Experiences",
                "pt": "Visitas e Experiências",
                "en_examples": "Guided tours, classic Lisbon experiences, attractions.",
                "pt_examples": "Visitas guiadas, experiências clássicas de Lisboa e atrações.",
            },
            {
                "keys": {"view points", "nature", "gardens & parks", "parks"},
                "emoji": "🌅",
                "en": "Viewpoints & Nature",
                "pt": "Miradouros e Natureza",
                "en_examples": "Viewpoints, parks, gardens, open-air places.",
                "pt_examples": "Miradouros, parques, jardins e espaços ao ar livre.",
            },
            {
                "keys": {"restaurants", "restaurant"},
                "emoji": "🍽️",
                "en": "Food & Restaurants",
                "pt": "Restaurantes e Gastronomia",
                "en_examples": "Restaurants and food-focused places.",
                "pt_examples": "Restaurantes e locais ligados à gastronomia.",
            },
            {
                "keys": {"shopping", "tourist offices"},
                "emoji": "🛍️",
                "en": "Shopping & Visitor Services",
                "pt": "Compras e Apoio ao Visitante",
                "en_examples": "Shopping areas and tourist information offices.",
                "pt_examples": "Zonas de compras e postos de informação turística.",
            },
            {
                "keys": {"tejo cruises"},
                "emoji": "⛵",
                "en": "River Cruises",
                "pt": "Cruzeiros no Tejo",
                "en_examples": "Tagus river cruise options.",
                "pt_examples": "Opções de cruzeiros no rio Tejo.",
            },
        ]
        output_parts = [f"### 🏛️ **{title}**\n"]

        # Count places by category
        for place in places_data:
            cat = place.get('category', 'Uncategorized')
            category_counts[cat] = category_counts.get(cat, 0) + 1

        normalized_counts = {
            _normalize_lookup_text(cat): count
            for cat, count in category_counts.items()
            if _normalize_lookup_text(cat) not in {"uncategorized", "dmcs & pcos", "18 holes"}
        }

        covered_keys: set[str] = set()
        for group in category_groups:
            count = sum(normalized_counts.get(key, 0) for key in group["keys"])
            if count <= 0:
                continue
            covered_keys.update(group["keys"])
            noun = place_word if count == 1 else place_word_plural
            label = group["pt"] if is_pt else group["en"]
            examples = group["pt_examples"] if is_pt else group["en_examples"]
            output_parts.append(f"- {group['emoji']} **{label}:** {count} {noun}")
            output_parts.append(f"    - {examples}")

        remaining_count = sum(
            count for key, count in normalized_counts.items()
            if key not in covered_keys and count > 0
        )
        if remaining_count:
            label = "Outras categorias especializadas" if is_pt else "Other specialist categories"
            output_parts.append(f"- 📌 **{label}:** {remaining_count} {place_word_plural}")

        total_label = "Total de locais" if is_pt else "Total places"
        tip = (
            "💡 **Dica:** Podes perguntar por uma categoria específica para receber locais concretos e verificados."
            if is_pt
            else "💡 **Tip:** Ask for a specific category to get concrete, verified places."
        )
        output_parts.append(f"\n📊 **{total_label}:** {len(places_data)}")
        output_parts.append(f"\n{tip}")

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
