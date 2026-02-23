# ==========================================================================
# Master Thesis - Web Knowledge Tool
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Real-time web search capability for the Researcher Agent.
#   Implements a "Search Waterfall" strategy:
#     1. Tavily Search (Optimized for AI, requires API Key)
#     2. DuckDuckGo Search (Free fallback)
#     3. Wikipedia (Encyclopedia fallback for history/facts)
# ==========================================================================

import logging
import os
from typing import Optional

import wikipedia
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool

# Configure logging
logger = logging.getLogger(__name__)

# Set Wikipedia language default to Portuguese
wikipedia.set_lang("pt")

# Constants
DEFAULT_SEARCH_RESULTS_TAVILY = 5  # Increased for completeness


def _search_tavily(query: str) -> Optional[str]:
    """
    Execute search using Tavily API.
    Returns comprehensive results without truncation.

    Args:
        query (str): The topic to search (e.g., "História do Castelo de São Jorge").
    
    Returns:
        str: Comprehensive search results from the best available source.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.debug("Tavily API key not found. Skipping.")
        return None

    try:
        # Initialize Tavily with more results (5-10 represents a good comprehensive set)
        tavily_tool = TavilySearchResults(max_results=DEFAULT_SEARCH_RESULTS_TAVILY)
        results = tavily_tool.invoke({"query": query})

        if not results:
            return None

        output = [f"🌐 **Resultados Tavily Search para '{query}':**"]
        
        # Format results nicely
        for res in results:
            url = res.get('url', 'N/A')
            content = res.get('content', '')
            # Append full content without truncation
            output.append(f"\n### 🔗 [{url}]({url})\n{content}")
        
        return "\n".join(output)

    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return None


def _search_duckduckgo(query: str) -> Optional[str]:
    """
    Execute search using DuckDuckGo.
    Free fallback using the HTML backend for robustness.

    Args:
        query (str): The topic to search (e.g., "História do Castelo de São Jorge").
    
    Returns:
        str: Comprehensive search results from the best available source.
    """
    try:
        # standardizing usage via wrapper for clarity/control
        wrapper = DuckDuckGoSearchAPIWrapper(backend="html")
        ddg_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
        
        result = ddg_tool.invoke(query)

        if not result:
            return None

        # DuckDuckGo result is usually a string of snippets.
        # We return it fully.
        return f"🦆 **Resultados DuckDuckGo para '{query}':**\n\n{result}"

    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return None


def _search_wikipedia(query: str, language: str = "pt") -> Optional[str]:
    """
    Execute search using Wikipedia.
    Returns the full summary/intro without forced sentence limits.

    Args:
        query (str): The topic to search (e.g., "História do Castelo de São Jorge").
        language (str): 'pt' for Portuguese, 'en' for English. Default 'pt'.
    
    Returns:
        str: Comprehensive search results from the best available source.
    """
    try:
        wikipedia.set_lang(language)
        # Search for the page matching the query
        search_results = wikipedia.search(query, results=1)
        
        if not search_results:
            return None
            
        page_title = search_results[0]
        
        # Get the page object
        page = wikipedia.page(page_title, auto_suggest=False)
        
        # Use content summary. 
        # Note: 'summary' usually retrieves the intro section. 
        # We avoid 'sentences=...' to get the full intro.
        summary = page.summary
        
        output = f"""📚 **Wikipédia: {page.title}**
🔗 URL: {page.url}

{summary}
"""
        return output
    except Exception as e:
        logger.warning(f"Wikipedia search failed: {e}")
        return None

# ==========================================================================
# Exported Tool
# ==========================================================================


@tool
def search_history_culture(query: str, language: str = "pt") -> str:
    """
    Search the web for historical facts, cultural context, and live information.
    
    Uses a waterfall strategy to ensure the best possible result:
    1. **Tavily Search** (Best for AI, high quality, requires Key).
    2. **DuckDuckGo** (Broad fallback).
    3. **Wikipedia** (Encyclopedia fallback).
    
    Args:
        query (str): The topic to search (e.g., "História do Castelo de São Jorge").
        language (str): 'pt' for Portuguese, 'en' for English. Default 'pt'.
    
    Returns:
        str: Comprehensive search results from the best available source.
    """
    
    # Context enhancement: Ensure search is localized to Lisbon/Portugal if needed
    # (Kept from original logic as it fits the project scope)
    search_query = query
    if "Lisboa" not in query and "Portugal" not in query:
        search_query = f"{query} Lisboa Portugal"
    
    # 1. Try Tavily (Premium/Complete)
    result = _search_tavily(search_query)
    if result:
        return result
        
    # 2. Try DuckDuckGo (Free/Broad)
    logger.info(f"Falling back to DuckDuckGo for query: {query}")
    result = _search_duckduckgo(search_query)
    if result:
        return result

    # 3. Try Wikipedia (Encyclopedia)
    # Use original query to avoid search noise with "Lisboa Portugal" if the entity is well known
    logger.info(f"Falling back to Wikipedia for query: {query}")
    result = _search_wikipedia(query, language)
    if result:
        return result
        
    return f"❌ Não foi possível encontrar informações sobre: '{query}' nas fontes disponíveis (Tavily, DDG, Wiki)."

# ==========================================================================
# Test Block
# ==========================================================================


if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Web Knowledge Tool Test (No Truncation)\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    test_queries = [
        ("História do Castelo de São Jorge", "pt"),
        ("Greve Metro Lisboa", "pt"), 
    ]
    
    for q, lang in test_queries:
        print(f"\n\033[1m🔎 Testing Query:\033[0m '{q}' ({lang})")
        try:
            result = search_history_culture.invoke({"query": q, "language": lang})
            print("-" * 40)
            print(result)
            print("-" * 40)
            print(f"📊 Result Length: {len(result)} chars")
            
        except Exception as e:
            print(f"❌ Failed: {e}")

    print("\n\033[1;32m✅ Web knowledge tests completed!\033[0m")
