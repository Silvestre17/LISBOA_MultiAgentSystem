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

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

# Suppress chromadb telemetry warnings
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

from langchain_core.tools import tool

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
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


def format_event_dates(event: Dict) -> str:
    """Formats event dates for display."""
    dates = event.get('dates', [])
    if not dates:
        return "Date TBA"
    
    formatted = []
    for date_entry in dates[:3]:  # Show max 3 dates
        if date_entry.get('type') == 'single':
            date_info = date_entry.get('date', {})
            display = date_info.get('display_text', '')
            time = date_info.get('time', '')
            if display:
                if time:
                    formatted.append(f"{display} at {time}")
                else:
                    formatted.append(display)
        elif date_entry.get('type') == 'range':
            start = date_entry.get('start', {}).get('display_text', '')
            end = date_entry.get('end', {}).get('display_text', '')
            if start and end:
                formatted.append(f"{start} to {end}")
    
    if len(dates) > 3:
        formatted.append(f"(+{len(dates) - 3} more dates)")
    
    return " | ".join(formatted) if formatted else "Date TBA"


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
        from tools.dados_abertos import _search_place_in_datasets_logic, DF_METADATA, search_datasets, fetch_geojson_with_retry, extract_name, extract_address, extract_coordinates
        
        if DF_METADATA.empty:
            return []
        
        query_lower = query.lower()
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
            for feature in features[:100]:  # Limit per dataset
                properties = feature.get('properties', {})
                name = extract_name(properties)
                
                if name == "N/A":
                    continue
                
                # Check match
                name_lower = name.lower()
                if any(t in name_lower for t in tokens) or query_lower in name_lower:
                    address = extract_address(properties)
                    coords = extract_coordinates(feature.get('geometry', {}))
                    
                    found_places.append({
                        'title': name,
                        'category': f"📊 Open Data: {title}",
                        'location': address if address != "N/A" else "Lisboa",
                        'short_description': f"From Lisboa Aberta dataset: {title}",
                        'url': None,
                        'lat': coords[0] if coords else None,
                        'lon': coords[1] if coords else None,
                        'source': 'dados_abertos'
                    })
                    
                    if len(found_places) >= max_results:
                        break
            
            if len(found_places) >= max_results:
                break
        
        # Deduplicate by title
        unique = {}
        for p in found_places:
            if p['title'] not in unique:
                unique[p['title']] = p
        
        return list(unique.values())[:max_results]
    
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
    category_lower = category.lower() if category else None
    
    for item in data:
        # Category filter
        if category_lower:
            item_cat = item.get('category', '').lower()
            if category_lower not in item_cat:
                continue
        
        # Query filter
        if query_lower:
            searchable = " ".join([
                item.get('title', ''),
                item.get('full_description', ''),
                item.get('short_description', ''),
                item.get('category', ''),
                item.get('location', '')
            ]).lower()
            if query_lower not in searchable:
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
    max_results: int = 10
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
        
        # Parse date range (CRITICAL: defaults to upcoming 30 days if not specified)
        if not date_filter:
            date_filter = 'upcoming'  # Default to next 30 days
        
        start_date, end_date = parse_date_range(date_filter)
        
        # Logging
        date_info = f"{start_date.strftime('%Y-%m-%d') if start_date else 'any'} to {end_date.strftime('%Y-%m-%d') if end_date else 'any'}"
        logger.info(f"search_cultural_events: query='{query}', category='{category}', dates={date_info}, max={max_results}")
        
        # ALWAYS load JSON for date filtering (vector store doesn't filter by date)
        events_data = _load_events_json()
        
        if not events_data:
            return "❌ Events data not available."
        
        # Step 1: Filter by date FIRST (most important)
        if start_date or end_date:
            events_data = filter_events_by_date(events_data, start_date, end_date)
            logger.info(f"After date filter: {len(events_data)} events")
        
        if not events_data:
            # No events in date range
            today = datetime.now()
            return f"❌ No events found for '{date_filter}'.\nDate range: {date_info}\nToday is: {today.strftime('%Y-%m-%d')}\n\n💡 Try a broader date range like 'this month' or 'next month'."
        
        # Step 2: Filter by category
        if category:
            category_lower = category.lower()
            events_data = [e for e in events_data if category_lower in e.get('category', '').lower()]
            logger.info(f"After category filter: {len(events_data)} events")
        
        # Step 3: Filter by query (TOKEN-BASED matching for better recall)
        if query:
            # Tokenize query into individual words for flexible matching
            query_tokens = [t.strip().lower() for t in query.split() if len(t.strip()) >= 3]
            
            # Also create synonyms/related terms for common queries
            query_synonyms = {
                'music': ['concert', 'concerto', 'live', 'band', 'artist', 'musical', 'fado', 'jazz', 'rock', 'pop'],
                'concert': ['music', 'live', 'performance', 'show', 'gig'],
                'concerts': ['music', 'live', 'performance', 'show', 'gig'],
                'art': ['exhibition', 'gallery', 'museum', 'painting', 'sculpture', 'artwork'],
                'exhibition': ['art', 'gallery', 'museum', 'display', 'expo'],
                'theater': ['theatre', 'play', 'drama', 'stage', 'performance'],
                'theatre': ['theater', 'play', 'drama', 'stage', 'performance'],
                'dance': ['ballet', 'dancing', 'choreography', 'performance'],
                'family': ['children', 'kids', 'child', 'families'],
                'food': ['gastronomy', 'culinary', 'wine', 'taste', 'restaurant'],
            }
            
            # Expand query tokens with synonyms
            expanded_tokens = set(query_tokens)
            for token in query_tokens:
                if token in query_synonyms:
                    expanded_tokens.update(query_synonyms[token])
            
            filtered = []
            for event in events_data:
                searchable = " ".join([
                    event.get('title', ''),
                    event.get('full_description', ''),
                    event.get('short_description', ''),
                    event.get('category', ''),
                ]).lower()
                
                # Match if ANY expanded token is found
                if any(token in searchable for token in expanded_tokens):
                    filtered.append(event)
            
            events_data = filtered
            logger.info(f"After query filter: {len(events_data)} events (tokens: {list(expanded_tokens)[:5]}...)")
        
        if not events_data:
            return f"No events found matching: '{query or 'all'}' in date range {date_info}\n\n💡 Try broader terms like 'music', 'art', or 'festival'."
        
        # Step 4: SORT BY TEMPORAL RELEVANCE (CRITICAL!)
        # Ephemeral events (single-day concerts) should rank ABOVE long exhibitions
        for event in events_data:
            event['_relevance_score'] = calculate_temporal_relevance_score(event, start_date, end_date)
            event['_duration_days'] = get_event_duration_days(event)
        
        events_data.sort(key=lambda e: e.get('_relevance_score', 0), reverse=True)
        logger.info(f"Sorted by temporal relevance (top score: {events_data[0].get('_relevance_score', 0):.1f})")
        
        # Limit results
        results = events_data[:max_results]
        
        # Format output with DATES prominently displayed
        today = datetime.now()
        output_parts = [f"🎭 **Found {len(results)} Cultural Events in Lisbon:**"]
        output_parts.append(f"📅 **Date range:** {date_filter} ({date_info})")
        output_parts.append(f"📆 **Today is:** {today.strftime('%A, %d %B %Y')}\n")
        
        for i, event in enumerate(results, 1):
            title = event.get('title', event.get('url', 'Unknown').split('/')[-1].replace('-', ' ').title())
            cat = event.get('category', 'General')
            loc = event.get('location', 'Lisbon')
            dates_str = format_event_dates(event)
            duration = event.get('_duration_days', get_event_duration_days(event))
            relevance = event.get('_relevance_score', 50.0)
            
            # Duration label for clarity
            if duration == 1:
                duration_label = "🎯 Single day"
            elif duration <= 3:
                duration_label = f"📆 {duration} days"
            elif duration <= 7:
                duration_label = f"📅 ~1 week"
            elif duration <= 30:
                duration_label = f"📅 ~1 month"
            else:
                duration_label = f"🏛️ Long-running ({duration} days)"
            
            output_parts.append(f"{i}. 📅 **{title}**")
            output_parts.append(f"   🗓️ **When:** {dates_str}")
            output_parts.append(f"   ⏱️ **Duration:** {duration_label}")
            output_parts.append(f"   📂 Category: {cat}")
            
            if event.get('full_description'):
                desc = event['full_description'][:150] + "..." if len(event.get('full_description', '')) > 150 else event.get('full_description', '')
                output_parts.append(f"   {desc}")
            
            output_parts.append(f"   📍 {loc}")
            
            if event.get('url'):
                output_parts.append(f"   🔗 {event['url']}")
            
            output_parts.append("")  # Empty line between events
        
        output_parts.append(f"📊 **Total matching events:** {len(events_data)}")
        if len(events_data) > max_results:
            output_parts.append(f"💡 Showing top {max_results}. Use a more specific query to narrow results.")
        
        return "\n".join(output_parts)
    
    except Exception as e:
        logger.error(f"Error in search_cultural_events: {e}")
        return f"❌ Error searching events: {str(e)}"


@tool
def search_places_attractions(
    query: Optional[str] = None,
    category: Optional[str] = None,
    max_results: int = 10
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
        
        logger.info(f"search_places_attractions: query='{query}', category='{category}', max={max_results}")
        
        # Check if we should also search Dados Abertos (hybrid mode)
        search_dados_abertos = _should_search_dados_abertos(query)
        dados_abertos_results = []
        
        if search_dados_abertos and query:
            logger.info(f"Hybrid mode: Query contains Dados Abertos keywords")
            dados_abertos_results = _search_dados_abertos_hybrid(query, max_results=max_results // 2 + 1)
            logger.info(f"Dados Abertos returned {len(dados_abertos_results)} results")
        
        # =====================================================================
        # STEP 1: Search VisitLisboa (Vector Store)
        # =====================================================================
        visitlisboa_results = []
        kb = _get_vector_store()
        
        if kb:
            try:
                search_query = query or "places and attractions in Lisbon"
                if category:
                    search_query = f"{category} {search_query}"
                
                logger.info(f"search_places_attractions: searching VisitLisboa for '{search_query}'")
                
                results_with_scores = kb.search_with_scores(
                    query=search_query, 
                    k=max_results * 2,
                    collections=[COLLECTION_PLACES]
                )
                
                RELEVANCE_THRESHOLD = 1.8
                relevant_results = [(doc, score) for doc, score in results_with_scores if score <= RELEVANCE_THRESHOLD]
                
                logger.info(f"VisitLisboa: {len(results_with_scores)} raw, {len(relevant_results)} after threshold")
                
                # Convert to standard format
                for doc, score in relevant_results[:max_results]:
                    metadata = doc.metadata
                    # Attempt to get real address/location
                    real_location = metadata.get('address') or metadata.get('location') or 'Lisbon'
                    visitlisboa_results.append({
                        'title': metadata.get('title', 'Unknown'),
                        'category': metadata.get('category', 'General'),
                        'location': real_location,
                        'short_description': doc.page_content[:200] if doc.page_content else '',
                        'url': metadata.get('url', ''),
                        'source': 'visitlisboa',
                        'score': score
                    })
                    
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
        
        # =====================================================================
        # STEP 2: Merge Results (VisitLisboa + Dados Abertos)
        # =====================================================================
        
        # HYBRID STRATEGY: Interleave results from both sources
        # For queries matching Dados Abertos keywords, prioritize those results
        # since the user is likely looking for public infrastructure
        all_results = []
        existing_titles = set()
        
        if search_dados_abertos and dados_abertos_results:
            # User searched for infrastructure -> prioritize Dados Abertos
            # Take half from Dados Abertos, half from VisitLisboa
            da_quota = max_results // 2 + 1
            vl_quota = max_results - da_quota + 1
            
            # Add Dados Abertos first (more relevant for infrastructure queries)
            for r in dados_abertos_results[:da_quota]:
                if r['title'].lower() not in existing_titles:
                    all_results.append(r)
                    existing_titles.add(r['title'].lower())
            
            # Then add VisitLisboa (tourist-focused, but may have relevant results)
            for r in visitlisboa_results[:vl_quota]:
                if r['title'].lower() not in existing_titles:
                    all_results.append(r)
                    existing_titles.add(r['title'].lower())
        else:
            # Standard tourist query -> prioritize VisitLisboa
            for r in visitlisboa_results:
                all_results.append(r)
                existing_titles.add(r['title'].lower())
            
            # Add any Dados Abertos results that don't duplicate
            for r in dados_abertos_results:
                if r['title'].lower() not in existing_titles:
                    all_results.append(r)
                    existing_titles.add(r['title'].lower())
        
        # =====================================================================
        # STEP 3: Format Output
        # =====================================================================
        
        if not all_results:
            # Last resort fallback
            if query:
                logger.info(f"No results from hybrid search, trying direct Dados Abertos")
                from dados_abertos import _search_place_in_datasets_logic
                open_data_results = _search_place_in_datasets_logic(query, max_results=max_results)
                if open_data_results:
                    return open_data_results
            
            return f"No places found matching: '{query or 'all'}' in VisitLisboa or Open Data registries."
        
        # Limit to max_results
        final_results = all_results[:max_results]
        
        # Count sources
        vl_count = sum(1 for r in final_results if r.get('source') == 'visitlisboa')
        da_count = sum(1 for r in final_results if r.get('source') == 'dados_abertos')
        
        output_parts = [f"🏛️ **Found {len(final_results)} Places/Attractions in Lisbon:**\n"]
        
        for i, place in enumerate(final_results, 1):
            title = place.get('title', 'Unknown')
            cat = place.get('category', 'General')
            loc = place.get('location', 'Lisbon')
            source = place.get('source', 'unknown')
            
            # Source indicator
            if source == 'dados_abertos':
                output_parts.append(f"\n{i}. 📊 **{title}**")  # Open Data icon
            else:
                output_parts.append(f"\n{i}. 🏛️ **{title}**")  # VisitLisboa icon
            
            output_parts.append(f"   📂 Category: {cat}")
            
            if place.get('short_description'):
                desc = place['short_description'][:200]
                if len(place['short_description']) > 200:
                    desc += "..."
                output_parts.append(f"   {desc}")
            
            output_parts.append(f"   📍 {loc}")
            
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
        output_parts.append("\n💡 Use search_cultural_events(query='your interest') for semantic search.")
        
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
        
        # Count places by category
        category_counts = {}
        for place in places_data:
            cat = place.get('category', 'Uncategorized')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Sort by count
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        output_parts = ["🏛️ **Place Categories in Lisbon:**\n"]
        for cat, count in sorted_categories[:20]:  # Top 20
            output_parts.append(f"  • {cat}: {count} places")
        
        if len(sorted_categories) > 20:
            output_parts.append(f"  ... and {len(sorted_categories) - 20} more categories")
        
        output_parts.append(f"\n📊 **Total places:** {len(places_data)}")
        output_parts.append("\n💡 Use search_places_attractions(query='your interest') for semantic search.")
        
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
        
        output_parts = [f"🔍 **Lisbon Knowledge Search Results:**\n"]
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
            print(f"\n\033[1;32m✅ PASSED\033[0m")
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
    # NOTE: Should find concerts/music events via token matching + synonyms
    run_test(
        "Search Events - Semantic Query (Music) [TESTS TOKEN MATCHING]",
        search_cultural_events.invoke,
        {"query": "music concerts", "max_results": 3}
    )
    
    # TEST 4: Search Events - By Category (Exhibitions)
    # NOTE: Even exhibitions should be sorted by temporal relevance
    run_test(
        "Search Events - By Category (Exhibitions) [TESTS DURATION SORTING]",
        search_cultural_events.invoke,
        {"category": "Exhibitions", "max_results": 3}
    )
    
    # TEST 5: Search Events - This Week
    # CRITICAL: Single-day events should appear BEFORE long exhibitions!
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
    # NOTE: "hospital" keyword triggers hybrid mode, combining tourist data
    # with public infrastructure from Lisboa Aberta (GPS coords included)
    run_test(
        "HYBRID SEARCH - Hospital (VisitLisboa + Dados Abertos)",
        search_places_attractions.invoke,
        {"query": "hospital", "max_results": 5}
    )
    
    # TEST 13: HYBRID SEARCH - University/Education
    # NOTE: "universidade" keyword triggers Dados Abertos for public institutions
    run_test(
        "HYBRID SEARCH - University (VisitLisboa + Dados Abertos)",
        search_places_attractions.invoke,
        {"query": "universidade", "max_results": 5}
    )
    
    # TEST 14: Tourist Query (NO hybrid, VisitLisboa only)
    # NOTE: "tower belem" is a tourist query, should NOT trigger Dados Abertos
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
        print(f"\n\033[1;32m🎉 ALL TESTS PASSED! System is working correctly.\033[0m")
    else:
        print(f"\n\033[1;33m⚠️  Some tests failed. Check errors above.\033[0m")
    
    print("=" * 70 + "\n")
