# ==========================================================================
# Master Thesis - VisitLisboa Semantic Search Tools
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Semantic search tools for VisitLisboa data using ChromaDB vector store.
#   Features:
#     - Semantic search over events using embeddings
#     - Semantic search over places/attractions using embeddings
#     - Category filtering with semantic understanding
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
from typing import Optional, List, Dict, Any

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
    max_results: int = 10
) -> str:
    """
    Search for cultural events in Lisbon using semantic search.
    
    This tool uses AI-powered semantic search to find relevant events based on
    meaning, not just keyword matching. It searches the VisitLisboa events database.
    
    Args:
        query (str, optional): Natural language query describing what you're looking for.
            Examples: 'music concerts', 'art exhibitions', 'family activities',
                     'outdoor events', 'Christmas celebrations', 'fado music'.
        category (str, optional): Filter by event category. Options include:
            'Main Events', 'Exhibitions', 'Music', 'Theater', 'Dance',
            'Cinema', 'Sports', 'Fairs', 'Festivals', 'Gastronomy'.
        max_results (int): Maximum number of results to return (default: 10).
    
    Returns:
        str: Formatted list of matching events with descriptions and links.
    
    Examples:
        - search_cultural_events(query="jazz concerts") -> Jazz and music events
        - search_cultural_events(category="Exhibitions") -> Art exhibitions
        - search_cultural_events(query="things to do with kids") -> Family events
    """
    try:
        # Normalize inputs
        query = str(query).strip() if query and str(query).strip() and str(query).lower() != 'none' else None
        category = str(category).strip() if category and str(category).strip() and str(category).lower() != 'none' else None
        
        if not isinstance(max_results, int) or max_results <= 0:
            max_results = 10
        
        logger.info(f"search_cultural_events: query='{query}', category='{category}', max={max_results}")
        
        # Try vector store first
        kb = _get_vector_store()
        
        if kb:
            # Build search query
            search_query = query or "cultural events in Lisbon"
            if category:
                search_query = f"{category} {search_query}"
            
            # Semantic search in events collection only
            results = kb.search(
                query=search_query, 
                k=max_results, 
                collections=[COLLECTION_EVENTS]
            )
            
            if not results:
                return f"No events found matching: '{query or 'all'}'"
            
            # Format output
            output_parts = [f"🎭 **Found {len(results)} Cultural Events in Lisbon:**\n"]
            
            for i, doc in enumerate(results, 1):
                output_parts.append(f"\n{i}. {_extract_event_from_doc(doc)}")
            
            output_parts.append(f"\n\n📊 **Search method:** Semantic (AI-powered)")
            output_parts.append("💡 Try more specific queries for better results.")
            
            return "\n".join(output_parts)
        
        else:
            # Fallback to JSON search
            logger.info("Using JSON fallback for events search")
            events_data = _load_events_json()
            
            if not events_data:
                return "❌ Events data not available."
            
            results = _fallback_search(query, category, events_data, max_results)
            
            if not results:
                return f"No events found matching: '{query or 'all'}'"
            
            output_parts = [f"🎭 **Found {len(results)} Cultural Events (keyword search):**\n"]
            
            for i, event in enumerate(results, 1):
                title = event.get('title', event.get('url', 'Unknown').split('/')[-1].replace('-', ' ').title())
                cat = event.get('category', 'General')
                loc = event.get('location', 'Lisbon')
                output_parts.append(f"\n{i}. 📅 **{title}**")
                output_parts.append(f"   Category: {cat}")
                if event.get('full_description'):
                    desc = event['full_description'][:200] + "..." if len(event.get('full_description', '')) > 200 else event.get('full_description', '')
                    output_parts.append(f"   {desc}")
                output_parts.append(f"   📍 {loc}")
            
            output_parts.append(f"\n\n📊 **Total events in database:** {len(events_data)}")
            
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
            # Build search query
            search_query = query or "places and attractions in Lisbon"
            if category:
                search_query = f"{category} {search_query}"
            
            # Semantic search in places collection only
            results = kb.search(
                query=search_query, 
                k=max_results, 
                collections=[COLLECTION_PLACES]
            )
            
            if not results:
                return f"No places found matching: '{query or 'all'}'"
            
            # Format output
            output_parts = [f"🏛️ **Found {len(results)} Places/Attractions in Lisbon:**\n"]
            
            for i, doc in enumerate(results, 1):
                output_parts.append(f"\n{i}. {_extract_place_from_doc(doc)}")
            
            output_parts.append(f"\n\n📊 **Search method:** Semantic (AI-powered)")
            output_parts.append("💡 Try more specific queries for better results.")
            
            return "\n".join(output_parts)
        
        else:
            # Fallback to JSON search
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
        
        # Search all collections
        results = kb.search(query=query, k=max_results)
        
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
