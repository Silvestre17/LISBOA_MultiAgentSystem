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

import os
import logging
import wikipedia
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

# Set Wikipedia language to Portuguese (default) or English
wikipedia.set_lang("pt")

# Constants for Token Optimization (approx chars)
MAX_CHARS_PER_RESULT = 350
MAX_RESULTS_TAVILY = 3
MAX_CHARS_TOTAL = 2000  # Hard limit for LLM context safety

# ==========================================================================
# Search Providers (Internal)
# ==========================================================================

def _truncate_content(content: str, limit: int = MAX_CHARS_TOTAL) -> str:
    """Helper to ensure we never blow up the context window."""
    if len(content) <= limit:
        return content
    return content[:limit] + "... [TRUNCATED]"

def _search_tavily(query: str, max_results: int = MAX_RESULTS_TAVILY) -> Optional[str]:
    """
    Execute search using Tavily API.
    Best for live events, news, and complex queries.
    """
    try:
        if not os.getenv("TAVILY_API_KEY"):
            # Only warn once to avoid log spam, or use debug level
            logger.debug("Tavily API key not found. Skipping.")
            return None
            
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        
        if not results:
            return None
            
        # Format results optimally for the LLM
        output = "🌐 **Resultados Tavily Search:**\n"
        for res in results:
            url = res.get('url', 'No URL')
            content = res.get('content', '')
            # Enforce strict char limit per result
            content_preview = content[:MAX_CHARS_PER_RESULT] + ("..." if len(content) > MAX_CHARS_PER_RESULT else "")
            output += f"- **{url}**: {content_preview}\n"
            
        return _truncate_content(output)
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return None


def _search_duckduckgo(query: str) -> Optional[str]:
    """
    Execute search using DuckDuckGo.
    Good free fallback for general web queries.
    """
    try:
        tool = DuckDuckGoSearchRun()
        result = tool.invoke(query)
        if result:
            # DuckDuckGo often returns a huge text blob, strict truncate needed
            formatted = f"🦆 **Resultados DuckDuckGo:**\n{result}"
            return _truncate_content(formatted, limit=1500)
        return None
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return None


def _search_wikipedia(query: str, language: str = "pt") -> Optional[str]:
    """
    Execute search using Wikipedia.
    Reliable for history, monuments, and biographies.
    """
    try:
        wikipedia.set_lang(language)
        search_results = wikipedia.search(query, results=1)
        
        if not search_results:
            return None
            
        page_title = search_results[0]
        # Fetch summary (limit sentences to keep context focused)
        summary = wikipedia.summary(page_title, sentences=5)
        url = wikipedia.page(page_title).url
        
        output = f"""📚 **Wikipédia (**{page_title}**)**
{summary}

🔗 Fonte: {url}
"""
        return _truncate_content(output)
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
    
    Uses a waterfall strategy:
    1. **Tavily Search** (Best for AI, requires Key).
    2. **DuckDuckGo** (Free fallback).
    3. **Wikipedia** (Encyclopedia fallback).
    
    Args:
        query (str): The topic to search (e.g., "História do Castelo de São Jorge", "Greve metro lisboa").
        language (str): 'pt' for Portuguese, 'en' for English. Default 'pt'.
    
    Returns:
        str: Summary of findings from the best available source.
    """
    
    # Context enhancement: Ensure search is localized to Lisbon/Portugal if needed
    search_query = f"{query} Lisboa Portugal" if "Lisboa" not in query and "Portugal" not in query else query
    
    # 1. Try Tavily
    result = _search_tavily(search_query)
    if result:
        return result
        
    # 2. Try DuckDuckGo
    logger.info(f"Falling back to DuckDuckGo for query: {query}")
    result = _search_duckduckgo(search_query)
    if result:
        return result

    # 3. Try Wikipedia (Use original query without "Lisboa Portugal" appendix for better title match)
    logger.info(f"Falling back to Wikipedia for query: {query}")
    result = _search_wikipedia(query, language)
    if result:
        return result
        
    return f"❌ Não encontrei informações sobre: '{query}' em nenhuma fonte (Tavily, DDG, Wiki)."


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Web Knowledge Tool Test (Waterfall + Token Safe)\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    test_queries = [
        ("História do Castelo de São Jorge", "pt"),
        ("Greve Metro Lisboa", "pt"), # Good for live search
        ("Who built Jeronimos Monastery?", "en")
    ]
    
    for q, lang in test_queries:
        print(f"\n\033[1m🔎 Testing Query:\033[0m '{q}' ({lang})")
        try:
            result = search_history_culture.invoke({"query": q, "language": lang})
            char_len = len(result)
            est_tokens = char_len // 4
            
            print("-" * 40)
            print(result[:500] + "..." if len(result) > 500 else result)
            print("-" * 40)
            print(f"📊 Stats: {char_len} chars (~{est_tokens} tokens)")
            if char_len > MAX_CHARS_TOTAL:
                 print("⚠️ WARNING: Result exceeds safe context limit!")
            else:
                 print("✅ Context Usage: OK")
            
        except Exception as e:
            print(f"❌ Failed: {e}")

    print("\n\033[1;32m✅ Web knowledge tests completed!\033[0m")
