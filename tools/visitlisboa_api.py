# ==========================================================================
# Master Thesis - VisitLisboa Data Tools
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Search tools for VisitLisboa scraped data (events and places).
#   Features:
#     - Cultural events search by keyword, category, or date
#     - Places/attractions search by keyword or category
#     - Daily updated data from VisitLisboa webscraping
# 
#   Data Sources:
#     - events.json: Cultural events, exhibitions, festivals
#     - places.json: Museums, monuments, restaurants, attractions
# ==========================================================================

# Required libraries:
# pip install langchain-core

import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from langchain_core.tools import tool

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================================================
# Data Loading
# ==========================================================================

def load_events() -> List[Dict[str, Any]]:
    """
    Loads events data from the VisitLisboa JSON file.
    
    Returns:
        List[Dict]: List of event dictionaries.
    """
    try:
        with open(Config.PATH_VISIT_LISBOA_EVENTS, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"✅ Loaded {len(data)} events from VisitLisboa")
            return data
    except FileNotFoundError:
        logger.warning(f"⚠️ Events file not found: {Config.PATH_VISIT_LISBOA_EVENTS}")
        return []
    except Exception as e:
        logger.error(f"❌ Error loading events: {e}")
        return []


def load_places() -> List[Dict[str, Any]]:
    """
    Loads places data from the VisitLisboa JSON file.
    
    Returns:
        List[Dict]: List of place dictionaries.
    """
    try:
        with open(Config.PATH_VISIT_LISBOA_PLACES, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"✅ Loaded {len(data)} places from VisitLisboa")
            return data
    except FileNotFoundError:
        logger.warning(f"⚠️ Places file not found: {Config.PATH_VISIT_LISBOA_PLACES}")
        return []
    except Exception as e:
        logger.error(f"❌ Error loading places: {e}")
        return []


# Load data once at module import
EVENTS_DATA = load_events()
PLACES_DATA = load_places()


# ==========================================================================
# Helper Functions
# ==========================================================================

def search_in_text(query: str, text: str) -> bool:
    """
    Case-insensitive search for query in text.
    
    Args:
        query (str): Search query.
        text (str): Text to search in.
        
    Returns:
        bool: True if query found in text.
    """
    if not text:
        return False
    return query.lower() in text.lower()


def extract_event_info(event: Dict[str, Any]) -> str:
    """
    Extracts a formatted summary from an event dictionary.
    
    Args:
        event (Dict): Event data.
        
    Returns:
        str: Formatted event summary.
    """
    parts = []
    
    # Title (from URL)
    if 'url' in event:
        title = event['url'].split('/')[-1].replace('-', ' ').title()
        parts.append(f"📅 **{title}**")
    
    # Category
    if event.get('category'):
        parts.append(f"   Category: {event['category']}")
    
    # Description
    if event.get('full_description'):
        desc = event['full_description'][:300]
        if len(event['full_description']) > 300:
            desc += "..."
        parts.append(f"   {desc}")
    
    # Dates
    if event.get('dates') and len(event['dates']) > 0:
        dates_str = ", ".join(event['dates'][:3])
        if len(event['dates']) > 3:
            dates_str += f" (+{len(event['dates']) - 3} more)"
        parts.append(f"   Dates: {dates_str}")
    
    # Location
    if event.get('location'):
        parts.append(f"   📍 Location: {event['location']}")
    
    # URL
    if event.get('url'):
        parts.append(f"   🔗 More info: {event['url']}")
    
    return "\n".join(parts)


def extract_place_info(place: Dict[str, Any]) -> str:
    """
    Extracts a formatted summary from a place dictionary.
    
    Args:
        place (Dict): Place data.
        
    Returns:
        str: Formatted place summary.
    """
    parts = []
    
    # Title
    if place.get('title'):
        parts.append(f"🏛️ **{place['title']}**")
    
    # Category
    if place.get('category'):
        parts.append(f"   Category: {place['category']}")
    
    # Short description
    if place.get('short_description'):
        parts.append(f"   {place['short_description']}")
    
    # Location
    if place.get('location'):
        parts.append(f"   📍 Location: {place['location']}")
    
    # Schedule (today)
    if place.get('schedule', {}).get('today'):
        parts.append(f"   🕐 {place['schedule']['today']}")
    
    # TripAdvisor rating
    if place.get('tripadvisor', {}).get('rating'):
        rating = place['tripadvisor']['rating']
        reviews = place['tripadvisor'].get('reviews_count', '?')
        parts.append(f"   ⭐ TripAdvisor: {rating}/5 ({reviews} reviews)")
    
    # Contact
    contact = place.get('contact_info', {})
    if contact.get('website'):
        parts.append(f"   🔗 Website: {contact['website']}")
    
    return "\n".join(parts)


# ==========================================================================
# LangChain Tools
# ==========================================================================

@tool
def search_cultural_events(
    query: Optional[str] = None,
    category: Optional[str] = None,
    max_results: int = 10
) -> str:
    """
    Search for cultural events in Lisbon from VisitLisboa data.
    
    Use this tool to find exhibitions, festivals, concerts, theater shows,
    and other cultural events happening in the Lisbon area.
    
    Args:
        query (str, optional): Search keyword (e.g., 'music', 'art', 'theater').
        category (str, optional): Event category filter. Options include:
            'Main Events', 'Exhibitions', 'Music', 'Theater', 'Dance',
            'Cinema', 'Sports', 'Fairs', 'Festivals'.
        max_results (int): Maximum number of results to return (default: 10).
    
    Returns:
        str: Formatted list of matching events with details.
    
    Examples:
        - search_cultural_events(query="jazz") -> Jazz concerts
        - search_cultural_events(category="Exhibitions") -> Art exhibitions
        - search_cultural_events(query="christmas") -> Christmas events
    """
    try:
        # Normalize inputs (handle None, empty strings, invalid types)
        query = str(query).strip() if query and str(query).strip() and str(query).lower() != 'none' else None
        category = str(category).strip() if category and str(category).strip() and str(category).lower() != 'none' else None
        
        # Ensure max_results is valid
        if not isinstance(max_results, int) or max_results <= 0:
            max_results = 10
        
        logger.info(f"search_cultural_events called: query={query}, category={category}, max_results={max_results}")
        
        if not EVENTS_DATA:
            return "❌ Events data not available. Please check the data files."
        
        results = []
        
        for event in EVENTS_DATA:
            # Filter by category
            if category:
                event_category = event.get('category', '')
                if not search_in_text(category, event_category):
                    continue
            
            # Filter by query (search in description, URL, category)
            if query:
                searchable_text = " ".join([
                    event.get('full_description', ''),
                    event.get('url', ''),
                    event.get('category', ''),
                    event.get('location', '')
                ])
                if not search_in_text(query, searchable_text):
                    continue
            
            results.append(event)
            
            if len(results) >= max_results:
                break
        
        if not results:
            search_terms = []
            if query:
                search_terms.append(f"query='{query}'")
            if category:
                search_terms.append(f"category='{category}'")
            return f"No events found matching: {', '.join(search_terms) if search_terms else 'all events'}"
        
        # Format output
        output_parts = [f"🎭 **Found {len(results)} Cultural Events in Lisbon:**\n"]
        
        for i, event in enumerate(results, 1):
            output_parts.append(f"\n{i}. {extract_event_info(event)}")
        
        # Add available categories hint
        categories = set(e.get('category', 'Unknown') for e in EVENTS_DATA if e.get('category'))
        output_parts.append(f"\n\n📌 **Available categories:** {', '.join(sorted(categories))}")
        output_parts.append(f"\n📊 **Total events in database:** {len(EVENTS_DATA)}")
        
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
    Search for places and attractions in Lisbon from VisitLisboa data.
    
    Use this tool to find museums, monuments, restaurants, viewpoints,
    and other tourist attractions in the Lisbon area.
    
    Args:
        query (str, optional): Search keyword (e.g., 'museum', 'restaurant', 'beach').
        category (str, optional): Place category filter. Options include:
            'Museums & Monuments', 'Restaurants', 'Hotels', 'View Points',
            'Beaches', 'Shopping', 'Nightlife', 'Parks & Gardens'.
        max_results (int): Maximum number of results to return (default: 10).
    
    Returns:
        str: Formatted list of matching places with details.
    
    Examples:
        - search_places_attractions(query="tower") -> Tower of Belém, etc.
        - search_places_attractions(category="Museums") -> All museums
        - search_places_attractions(query="fado") -> Fado houses
    """
    try:
        # Normalize inputs (handle None, empty strings, invalid types)
        query = str(query).strip() if query and str(query).strip() and str(query).lower() != 'none' else None
        category = str(category).strip() if category and str(category).strip() and str(category).lower() != 'none' else None
        
        # Ensure max_results is valid
        if not isinstance(max_results, int) or max_results <= 0:
            max_results = 10
        
        logger.info(f"search_places_attractions called: query={query}, category={category}, max_results={max_results}")
        
        if not PLACES_DATA:
            return "❌ Places data not available. Please check the data files."
        
        results = []
        
        for place in PLACES_DATA:
            # Filter by category
            if category:
                place_category = place.get('category', '')
                if not search_in_text(category, place_category):
                    continue
            
            # Filter by query (search in title, description, category, location)
            if query:
                searchable_text = " ".join([
                    place.get('title', ''),
                    place.get('short_description', ''),
                    place.get('full_description', ''),
                    place.get('category', ''),
                    place.get('location', '')
                ])
                if not search_in_text(query, searchable_text):
                    continue
            
            results.append(place)
            
            if len(results) >= max_results:
                break
        
        if not results:
            search_terms = []
            if query:
                search_terms.append(f"query='{query}'")
            if category:
                search_terms.append(f"category='{category}'")
            return f"No places found matching: {', '.join(search_terms) if search_terms else 'all places'}"
        
        # Format output
        output_parts = [f"🏛️ **Found {len(results)} Places/Attractions in Lisbon:**\n"]
        
        for i, place in enumerate(results, 1):
            output_parts.append(f"\n{i}. {extract_place_info(place)}")
        
        # Add available categories hint
        categories = set(p.get('category', 'Unknown') for p in PLACES_DATA if p.get('category'))
        output_parts.append(f"\n\n📌 **Available categories:** {', '.join(sorted(list(categories)[:15]))}...")
        output_parts.append(f"\n📊 **Total places in database:** {len(PLACES_DATA)}")
        
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
        
        if not EVENTS_DATA:
            return "❌ Events data not available."
        
        # Count events by category
        category_counts = {}
        for event in EVENTS_DATA:
            cat = event.get('category', 'Uncategorized')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Sort by count
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        output_parts = ["🎭 **Event Categories in Lisbon:**\n"]
        for cat, count in sorted_categories:
            output_parts.append(f"  • {cat}: {count} events")
        
        output_parts.append(f"\n📊 **Total events:** {len(EVENTS_DATA)}")
        output_parts.append("\n💡 Use search_cultural_events(category='Category Name') to filter.")
        
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
        
        if not PLACES_DATA:
            return "❌ Places data not available."
        
        # Count places by category
        category_counts = {}
        for place in PLACES_DATA:
            cat = place.get('category', 'Uncategorized')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Sort by count
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        output_parts = ["🏛️ **Place Categories in Lisbon:**\n"]
        for cat, count in sorted_categories[:20]:  # Top 20
            output_parts.append(f"  • {cat}: {count} places")
        
        if len(sorted_categories) > 20:
            output_parts.append(f"  ... and {len(sorted_categories) - 20} more categories")
        
        output_parts.append(f"\n📊 **Total places:** {len(PLACES_DATA)}")
        output_parts.append("\n💡 Use search_places_attractions(category='Category Name') to filter.")
        
        return "\n".join(output_parts)
    
    except Exception as e:
        logger.error(f"Error in get_place_categories: {e}")
        return f"❌ Error getting place categories: {str(e)}"


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 Testing VisitLisboa Tools")
    print("=" * 60)
    
    # Test event categories
    print("\n📋 Event Categories:")
    print(get_event_categories.invoke({}))
    
    # Test event search
    print("\n" + "-" * 40)
    print("🔍 Searching for 'exhibition' events:")
    print(search_cultural_events.invoke({"query": "exhibition", "max_results": 3}))
    
    # Test place categories
    print("\n" + "-" * 40)
    print("📋 Place Categories:")
    print(get_place_categories.invoke({}))
    
    # Test place search
    print("\n" + "-" * 40)
    print("🔍 Searching for 'museum' places:")
    print(search_places_attractions.invoke({"query": "museum", "max_results": 3}))
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
