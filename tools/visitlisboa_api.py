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
        
        # Step 3: Filter by query (keyword match)
        if query:
            query_lower = query.lower()
            filtered = []
            for event in events_data:
                searchable = " ".join([
                    event.get('title', ''),
                    event.get('full_description', ''),
                    event.get('short_description', ''),
                    event.get('category', ''),
                ]).lower()
                if query_lower in searchable:
                    filtered.append(event)
            events_data = filtered
            logger.info(f"After query filter: {len(events_data)} events")
        
        if not events_data:
            return f"No events found matching: '{query or 'all'}' in date range {date_info}"
        
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
            
            output_parts.append(f"{i}. 📅 **{title}**")
            output_parts.append(f"   🗓️ **When:** {dates_str}")
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
    Search for places and attractions in Lisbon using semantic search.
    
    This tool uses AI-powered semantic search to find relevant places based on
    meaning, not just keyword matching. It searches the VisitLisboa places database.
    
    Args:
        query (str, optional): Natural language query describing what you're looking for.
            Examples: 'museums with art', 'good restaurants for dinner',
                     'places to see sunset', 'historic monuments', 'beaches nearby'.
        category (str, optional): Filter by place category. Options include:
            'Museums & Monuments', 'Restaurants', 'Hotels', 'View Points',
            'Beaches', 'Shopping', 'Nightlife', 'Parks & Gardens', 'Tours'.
        max_results (int): Maximum number of results to return (default: 10).
    
    Returns:
        str: Formatted list of matching places with descriptions and links.
    
    Examples:
        - search_places_attractions(query="tower") -> Belém Tower, etc.
        - search_places_attractions(category="Museums") -> All museums
        - search_places_attractions(query="romantic dinner") -> Restaurants
    """
    try:
        # Normalize inputs
        query = str(query).strip() if query and str(query).strip() and str(query).lower() != 'none' else None
        category = str(category).strip() if category and str(category).strip() and str(category).lower() != 'none' else None
        
        if not isinstance(max_results, int) or max_results <= 0:
            max_results = 10
        
        logger.info(f"search_places_attractions: query='{query}', category='{category}', max={max_results}")
        
        # Try vector store first
        kb = _get_vector_store()
        
        if kb:
            try:
                # Build search query
                search_query = query or "places and attractions in Lisbon"
                if category:
                    search_query = f"{category} {search_query}"
                
                logger.info(f"search_places_attractions: searching for '{search_query}'")
                
                # Semantic search in places collection only
                results = kb.search(
                    query=search_query, 
                    k=max_results, 
                    collections=[COLLECTION_PLACES]
                )
                
                logger.info(f"search_places_attractions: found {len(results)} results")
                
                if results:
                    # Format output
                    output_parts = [f"🏛️ **Found {len(results)} Places/Attractions in Lisbon:**\n"]
                    
                    for i, doc in enumerate(results, 1):
                        output_parts.append(f"\n{i}. {_extract_place_from_doc(doc)}")
                    
                    output_parts.append(f"\n\n📊 **Search method:** Semantic (AI-powered)")
                    output_parts.append("💡 Try more specific queries for better results.")
                    
                    return "\n".join(output_parts)
                    
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to JSON: {e}")
        
        # Fallback to JSON search (if vector store fails or returns no results)
        logger.info("Using JSON fallback for places search")
        places_data = _load_places_json()
        
        if not places_data:
            return "❌ Places data not available."
        
        results = _fallback_search(query, category, places_data, max_results)
        
        if not results:
            return f"No places found matching: '{query or 'all'}'"
        
        output_parts = [f"🏛️ **Found {len(results)} Places/Attractions (keyword search):**\n"]
        
        for i, place in enumerate(results, 1):
            title = place.get('title', 'Unknown')
            cat = place.get('category', 'General')
            loc = place.get('location', 'Lisbon')
            output_parts.append(f"\n{i}. 🏛️ **{title}**")
            output_parts.append(f"   Category: {cat}")
            if place.get('short_description'):
                output_parts.append(f"   {place['short_description'][:200]}")
            output_parts.append(f"   📍 {loc}")
        
        output_parts.append(f"\n\n📊 **Total places in database:** {len(places_data)}")
        
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
    print("\n" + "=" * 60)
    print("🧪 Testing VisitLisboa Semantic Search Tools")
    print("=" * 60)
    
    # Test event categories
    print("\n📋 Event Categories:")
    print(get_event_categories.invoke({}))
    
    # Test event search
    print("\n" + "-" * 40)
    print("🔍 Searching for 'music' events (semantic):")
    print(search_cultural_events.invoke({"query": "music concerts", "max_results": 3}))
    
    # Test place categories
    print("\n" + "-" * 40)
    print("📋 Place Categories:")
    print(get_place_categories.invoke({}))
    
    # Test place search
    print("\n" + "-" * 40)
    print("🔍 Searching for 'historic' places (semantic):")
    print(search_places_attractions.invoke({"query": "historic monuments", "max_results": 3}))
    
    # Test comprehensive search
    print("\n" + "-" * 40)
    print("🔍 Searching across all knowledge bases:")
    print(search_lisbon_knowledge.invoke({"query": "Lisboa Card benefits"}))
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
