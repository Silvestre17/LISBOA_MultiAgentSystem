# ==========================================================================
# Master Thesis - Web Knowledge Tool
#   - André Filipe Gomes Silvestre, 20240502
#
#   Real-time web search capability for the Researcher Agent.
#   Implements a "Search Waterfall" strategy:
#     1. Tavily Search (Optimized for AI, requires API Key)
#     2. DuckDuckGo Search (Free fallback)
#     3. Wikipedia (Encyclopedia fallback for history/facts)
#
#   Usage:
#     > python tools/web_knowledge.py
#       Run the manual web-search fallback test queries for the Researcher agent.
# ==========================================================================

import logging
import os
import re
import warnings
from datetime import datetime
from typing import Optional
from urllib.parse import quote, urlparse
import sys

import requests
import wikipedia
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool

try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
except Exception:  # pragma: no cover - compatibility with older LangChain builds
    LangChainDeprecationWarning = Warning

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Configure logging
logger = logging.getLogger(__name__)

# Set Wikipedia language default to Portuguese
wikipedia.set_lang("pt")

# Constants
DEFAULT_SEARCH_RESULTS_TAVILY = 5  # Increased for completeness
LIVE_INFO_HINTS = {
    "now",
    "current",
    "latest",
    "live",
    "today",
    "tonight",
    "currently",
    "right now",
    "breaking",
    "news",
    "alert",
    "alerts",
    "strike",
    "strikes",
    "closure",
    "closed",
    "delay",
    "delays",
    "greve",
    "greves",
    "agora",
    "hoje",
    "neste momento",
    "últimas",
    "ultimas",
    "fechado",
    "fechada",
    "atraso",
    "atrasos",
}
LIVE_FALLBACK_DOMAINS = (
    "metrolisboa.pt",
    "carris.pt",
    "carrismetropolitana.pt",
    "cp.pt",
    "ipma.pt",
    "lisboa.pt",
    "cm-lisboa.pt",
    "visitlisboa.com",
)
LIVE_OPERATOR_DOMAIN_HINTS = {
    ("metro", "metropolitano", "metrolisboa", "metropolitano lisbon", "metro lisbo"): (
        "metrolisboa.pt",
        "cm-lisboa.pt",
    ),
    ("carris", "eléctrico", "eletrico", "tram", "autocarro", "carris urban"): (
        "carris.pt",
        "carrismetropolitana.pt",
        "lisboa.pt",
    ),
    ("carrimetropolitana", "carrismetropolitana", "carris metropolitana"): (
        "carrismetropolitana.pt",
        "carris.pt",
        "lisboa.pt",
    ),
    ("comboio", "cp", "sintra", "cascais", "azambuja"): (
        "cp.pt",
        "lisboa.pt",
    ),
    ("clima", "tempo", "meteo", "temperatura", "chuva"): (
        "ipma.pt",
    ),
}
AUTHORITATIVE_DOMAINS = (
    "wikipedia.org",
    "visitlisboa.com",
    "lisboa.pt",
    "cm-lisboa.pt",
    "gov.pt",
    "metrolisboa.pt",
    "carris.pt",
    "carrismetropolitana.pt",
    "cp.pt",
    "ipma.pt",
)
LIVE_FRESHNESS_DAYS = 30
_MONTH_MAP = {
    "jan": 1, "january": 1, "janeiro": 1,
    "feb": 2, "february": 2, "fev": 2, "fevereiro": 2,
    "mar": 3, "march": 3, "marco": 3, "março": 3,
    "apr": 4, "april": 4, "abr": 4, "abril": 4,
    "may": 5, "maio": 5,
    "jun": 6, "june": 6, "junho": 6,
    "jul": 7, "july": 7, "julho": 7,
    "aug": 8, "august": 8, "ago": 8, "agosto": 8,
    "sep": 9, "sept": 9, "september": 9, "set": 9, "setembro": 9,
    "oct": 10, "october": 10, "out": 10, "outubro": 10,
    "nov": 11, "november": 11, "novembro": 11,
    "dec": 12, "december": 12, "dez": 12, "dezembro": 12,
}


def _extract_domain(url: str) -> str:
    """Returns a normalized domain for a URL."""
    try:
        domain = urlparse(url).netloc.lower().strip()
    except Exception:
        return ""
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _is_authoritative_domain(url: str) -> bool:
    """Returns whether a URL belongs to a trusted knowledge domain."""
    domain = _extract_domain(url)
    if not domain:
        return False
    return any(domain == trusted or domain.endswith(f".{trusted}") for trusted in AUTHORITATIVE_DOMAINS)


def _is_live_info_query(query: str) -> bool:
    """Detects whether the query is about current/live information rather than background knowledge."""
    query_lower = (query or "").lower()
    return any(hint in query_lower for hint in LIVE_INFO_HINTS)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """Return list items in original order while removing duplicates."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _get_live_priority_domains(query: str) -> list[str]:
    """Build a prioritized list of official domains for live queries."""
    query_lower = (query or "").lower()
    domains: list[str] = []

    for keywords, mapped_domains in LIVE_OPERATOR_DOMAIN_HINTS.items():
        if any(keyword in query_lower for keyword in keywords):
            domains.extend(mapped_domains)

    if not domains:
        domains.extend(LIVE_FALLBACK_DOMAINS)

    return _dedupe_preserve_order(domains)


def _build_live_domain_queries(query: str) -> list[str]:
    """Build a list of live-focused, domain-scoped search queries."""
    normalized = query or ""
    base_query = normalized
    if "lisboa" not in normalized.lower() and "portugal" not in normalized.lower():
        base_query = f"{normalized} Lisboa Portugal"

    queries = [base_query]
    for domain in _get_live_priority_domains(normalized):
        queries.append(f"{base_query} site:{domain}")
    return _dedupe_preserve_order(queries)


def _build_no_live_data_message(language: str = "pt", query: str = "") -> str:
    """Build a strict no-result response for current/live queries."""
    if language == "en":
        return (
            f"⚠️ I could not find a reliable current-time source for: '{query}'."
            " For live disruptions, delays, and operational changes, verify on official channels before acting."
            "\n- Metro Lisboa: https://www.metrolisboa.pt/\n"
            "- Carris: https://www.carris.pt/\n"
            "- Carris Metropolitana: https://www.carrismetropolitana.pt/\n"
            "- CP: https://www.cp.pt/\n"
            "- IPMA: https://www.ipma.pt/"
        )

    return (
        f"⚠️ Não consegui confirmar uma fonte de referência recente para: '{query}'."
        " Para questões em tempo real (greves, atrasos, cortes de serviço), confirma a informação em canais oficiais antes de agir."
        "\n- Metro Lisboa: https://www.metrolisboa.pt/\n"
        "- Carris: https://www.carris.pt/\n"
        "- Carris Metropolitana: https://www.carrismetropolitana.pt/\n"
        "- CP: https://www.cp.pt/\n"
        "- IPMA: https://www.ipma.pt/"
    )


def _extract_result_datetime(text: str = "", url: str = "") -> Optional[datetime]:
    """Extracts a publication-like date from a snippet or URL when available."""
    haystacks = [text or "", url or ""]

    for haystack in haystacks:
        match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", haystack)
        if match:
            year, month, day = map(int, match.groups())
            try:
                return datetime(year, month, day)
            except ValueError:
                pass

        match = re.search(r"\b(\d{1,2})[./-](\d{1,2})[./-](20\d{2})\b", haystack)
        if match:
            day, month, year = map(int, match.groups())
            try:
                return datetime(year, month, day)
            except ValueError:
                pass

        match = re.search(r"\b([A-Za-zÀ-ÿ]{3,})\s+(\d{1,2}),\s*(20\d{2})\b", haystack)
        if match:
            month_text, day_text, year_text = match.groups()
            month = _MONTH_MAP.get(month_text.lower())
            if month:
                try:
                    return datetime(int(year_text), month, int(day_text))
                except ValueError:
                    pass

        match = re.search(r"\b(\d{1,2})\s+de\s+([A-Za-zÀ-ÿ]{3,})\s+de\s+(20\d{2})\b", haystack, re.IGNORECASE)
        if match:
            day_text, month_text, year_text = match.groups()
            month = _MONTH_MAP.get(month_text.lower())
            if month:
                try:
                    return datetime(int(year_text), month, int(day_text))
                except ValueError:
                    pass

    return None


def _has_recent_live_datetime_in_text(text: str = "") -> bool:
    """Check whether the text contains at least one recent date for live queries."""
    haystacks = [(text or "").replace(";", ".").split("\n")]
    for fragment_group in haystacks:
        for fragment in fragment_group:
            result_date = _extract_result_datetime(fragment)
            if result_date and _is_recent_live_result(result_date):
                return True
    return False


def _is_recent_live_result(result_date: Optional[datetime]) -> bool:
    """Returns whether a dated result is recent enough for live-information queries."""
    if result_date is None:
        return False
    age_days = (datetime.now() - result_date).days
    return 0 <= age_days <= LIVE_FRESHNESS_DAYS


def _search_wikipedia_with_fallback(query: str, language: str = "pt") -> Optional[str]:
    """Searches Wikipedia in the requested language and falls back to English when needed."""
    preferred = language if language in {"pt", "en"} else "pt"
    languages = [preferred]
    if preferred != "en":
        languages.append("en")

    for lang in languages:
        result = _search_wikipedia(query, lang)
        if result:
            if lang != preferred:
                note = (
                    "ℹ️ Source language fallback: no suitable result was found in the requested language, so the English Wikipedia article was used."
                    if preferred == "en"
                    else "ℹ️ Idioma de fallback da fonte: não foi encontrado um resultado fiável no idioma pedido, por isso foi usado o artigo da Wikipédia em inglês."
                )
                return f"{result}\n{note}"
            return result

    return None


def _get_low_confidence_notice(language: str = "pt") -> str:
    """Returns a caution notice for broad web-snippet fallbacks."""
    if language == "en":
        return (
            "⚠️ Broad web-snippet fallback: treat this as directional context and verify exact live details on official sources before acting on them."
        )
    return (
        "⚠️ Fallback com snippets genéricos da web: usa isto como contexto inicial e confirma detalhes exatos e em tempo real nas fontes oficiais antes de agir."
    )


def _get_live_result_notice(language: str = "pt") -> str:
    """Returns a short reminder for live/current web-result checks."""
    if language == "en":
        return (
            "⚠️ Live-information note: even official web results may describe previous disruptions or notices, so confirm the date and time on the linked source before assuming it still applies now."
        )
    return (
        "⚠️ Nota sobre informação em tempo real: até resultados de fontes oficiais podem referir interrupções ou avisos antigos, por isso confirma sempre a data e a hora na página ligada antes de assumir que ainda se aplicam agora."
    )


def _search_tavily(
    query: str,
    language: str = "pt",
    live_query: bool = False,
) -> tuple[Optional[str], bool]:
    """
    Execute search using Tavily API.
    Returns comprehensive results without truncation.

    Args:
        query (str): The topic to search (e.g., "História do Castelo de São Jorge").

    Returns:
        tuple[Optional[str], bool]: (search text, has_recent_live_signal).
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.debug("Tavily API key not found. Skipping.")
        return None, False

    try:
        # Initialize Tavily with more results (5-10 represents a good comprehensive set)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The class `TavilySearchResults` was deprecated.*",
                category=LangChainDeprecationWarning,
            )
            tavily_tool = TavilySearchResults(max_results=DEFAULT_SEARCH_RESULTS_TAVILY)
        results = tavily_tool.invoke({"query": query})

        if not results:
            return None, False

        enriched_results = []
        for res in results:
            url = res.get("url", "")
            content = res.get("content", "")
            result_date = _extract_result_datetime(content, url)
            enriched_results.append(
                {
                    **res,
                    "_result_date": result_date,
                    "_is_authoritative": _is_authoritative_domain(url),
                    "_is_recent": _is_recent_live_result(result_date),
                }
            )

        if live_query:
            filtered_results = []
            for res in enriched_results:
                if res["_is_authoritative"]:
                    filtered_results.append(res)
                    continue
                result_date = res["_result_date"]
                if result_date and result_date.year >= datetime.now().year and _is_recent_live_result(result_date):
                    filtered_results.append(res)
            if filtered_results:
                enriched_results = filtered_results

        sorted_results = sorted(
            enriched_results,
            key=lambda res: (
                0 if res["_is_authoritative"] else 1,
                0 if res["_is_recent"] else 1,
                -(res["_result_date"].timestamp()) if res["_result_date"] else float("inf"),
                _extract_domain(res.get("url", "")),
            ),
        )

        has_recent_live_signal = any(res["_is_recent"] for res in sorted_results)

        output = [f"🌐 **Resultados Tavily Search para '{query}':**"]
        if live_query and not has_recent_live_signal:
            if language == "en":
                output.append(
                    "⚠️ No recent dated official result was found in the retrieved web results. Treat older notices as historical context, not confirmation that a disruption is active right now."
                )
            else:
                output.append(
                    "⚠️ Não encontrei um resultado oficial recente com data nas fontes devolvidas. Trata avisos antigos como contexto histórico, não como confirmação de que a perturbação continua ativa agora."
                )

        # Format results nicely
        for res in sorted_results[:5]:
            url = res.get('url', 'N/A')
            content = res.get('content', '')
            domain = _extract_domain(url) or "web"
            source_badge = "✅ authoritative" if res["_is_authoritative"] else "🌍 web"
            if live_query:
                if res["_result_date"]:
                    freshness = "🟢 recent" if res["_is_recent"] else "🟠 older"
                    date_note = res["_result_date"].strftime("%Y-%m-%d")
                    source_badge = f"{source_badge} | {freshness} | {date_note}"
                else:
                    source_badge = f"{source_badge} | ⚪ undated"
            # Append full content without truncation
            output.append(f"\n### {source_badge} [{domain}]({url})\n{content}")

        return "\n".join(output), has_recent_live_signal

    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return None, False


def _search_duckduckgo(query: str, live_query: bool = False) -> tuple[Optional[str], bool]:
    """
    Execute search using DuckDuckGo.
    Free fallback using the HTML backend for robustness.

    Args:
        query (str): The topic to search (e.g., "História do Castelo de São Jorge").

    Returns:
        tuple[Optional[str], bool]: (search text, has_recent_live_signal).
    """
    try:
        # standardizing usage via wrapper for clarity/control
        wrapper = DuckDuckGoSearchAPIWrapper(backend="html")
        ddg_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)

        result = ddg_tool.invoke(query)

        if not result:
            return None, False

        result_text = f"[Resultados DuckDuckGo para '{query}']\n\n{result}"
        if not live_query:
            return result_text, False

        has_recent_live_signal = _has_recent_live_datetime_in_text(result_text)
        if not has_recent_live_signal:
            logger.info("DuckDuckGo live query did not return a recent dated signal: %s", query)
            return None, False

        # DuckDuckGo result is usually a string of snippets.
        # We return it fully when it includes a recent/live signal.
        return result_text, True

    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return None, False


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
    normalized_language = language if language in {"pt", "en"} else "pt"

    try:
        wikipedia.set_lang(normalized_language)
        search_results = wikipedia.search(query, results=1)
    except Exception as exc:
        logger.warning("Wikipedia library search failed for %r (%s): %s", query, normalized_language, exc)
        search_results = []

    if search_results:
        page_title = search_results[0]
        rest_result = _fetch_wikipedia_rest_summary(page_title, normalized_language)
        if rest_result:
            return rest_result

    try:
        wikipedia.set_lang(normalized_language)
        # Search for the page matching the query
        search_results = search_results or wikipedia.search(query, results=1)

        if not search_results:
            return None

        page_title = search_results[0]

        # Get the page object
        page = wikipedia.page(page_title, auto_suggest=False)

        # Use content summary.
        # Note: 'summary' usually retrieves the intro section.
        # We avoid 'sentences=...' to get the full intro.
        summary = page.summary

        safe_page_url = str(page.url or "")
        if "(" in safe_page_url or ")" in safe_page_url:
            try:
                from urllib.parse import urlparse, urlunparse
                parsed = urlparse(safe_page_url)
                encoded_path = parsed.path.replace("(", "%28").replace(")", "%29")
                safe_page_url = urlunparse(parsed._replace(path=encoded_path))
            except Exception:
                pass
        return f"""📚 **Wikipédia: {page.title}**
🔗 [Wikipedia]({safe_page_url})

{summary}
"""
    except Exception as e:
        logger.warning(f"Wikipedia search failed: {e}")
        return None


def _fetch_wikipedia_rest_summary(page_title: str, language: str = "pt") -> Optional[str]:
    """Fetch a Wikipedia summary through the stable REST endpoint.

    The ``wikipedia`` Python package sometimes fails when the legacy search API
    returns an HTML or empty body. The REST summary endpoint is simpler and keeps
    the history/culture waterfall useful when that library path is temporarily
    brittle.
    """
    normalized_language = language if language in {"pt", "en"} else "pt"
    title = (page_title or "").strip()
    if not title:
        return None

    url = f"https://{normalized_language}.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}"
    headers = {"User-Agent": "LISBOA-Thesis-Agent/1.0 (academic research; contact via repository)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning("Wikipedia REST summary failed for %r (%s): %s", title, normalized_language, exc)
        return None

    summary = str(payload.get("extract") or "").strip()
    resolved_title = str(payload.get("title") or title).strip()
    page_url = str(payload.get("content_urls", {}).get("desktop", {}).get("page") or "").strip()
    if not summary:
        return None
    page_url = page_url or f"https://{normalized_language}.wikipedia.org/wiki/{quote(resolved_title.replace(' ', '_'), safe='_')}"
    if "(" in page_url or ")" in page_url:
        try:
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(page_url)
            encoded_path = parsed.path.replace("(", "%28").replace(")", "%29")
            page_url = urlunparse(parsed._replace(path=encoded_path))
        except Exception:
            pass
    url_line = f"🔗 [Wikipedia]({page_url})"
    return f"📚 **Wikipédia: {resolved_title}**\n{url_line}\n\n{summary}\n"

# ==========================================================================
# Exported Tool
# ==========================================================================


@tool
def search_history_culture(query: str, language: str = "pt") -> str:
    """
    Search the web for historical facts, cultural context, and live information.

    Uses a quality-first strategy:
    - background/cultural queries -> Wikipedia -> Tavily -> DuckDuckGo
    - live queries -> Tavily (freshness-aware) -> scoped live DuckDuckGo -> official source guidance.

    Args:
        query (str): The topic to search (e.g., "História do Castelo de São Jorge").
        language (str): 'pt' for Portuguese, 'en' for English. Default 'pt'.

    Returns:
        str: Comprehensive search results from the best available source.
    """

    # Context enhancement: Ensure search is localized to Lisbon/Portugal if needed
    # (Kept from original logic as it fits the project scope)
    search_query = query
    if not re.search(r"\b(?:lisboa|lisbon|portugal)\b", query, flags=re.IGNORECASE):
        search_query = f"{query} Lisboa Portugal"

    live_query = _is_live_info_query(query)

    # Prefer encyclopedic sources for background history/culture.
    if not live_query:
        result = _search_wikipedia_with_fallback(query, language)
        if result:
            return result

    # Tavily remains the best general web option for current/live information.
    result, has_live_signal = _search_tavily(search_query, language=language, live_query=live_query)
    if result and (not live_query or has_live_signal):
        if live_query:
            return f"{result}\n\n{_get_live_result_notice(language)}"
        return result

    if live_query and not has_live_signal:
        logger.info(f"Attempting live domain-scoped fallback for query: {query}")
        for scoped_query in _build_live_domain_queries(query):
            if scoped_query == search_query:
                continue
            result, has_scoped_live_signal = _search_tavily(
                scoped_query,
                language=language,
                live_query=live_query,
            )
            if result and has_scoped_live_signal:
                return f"{result}\n\n{_get_live_result_notice(language)}"

    # Broad web search is kept as a fallback, but should be treated carefully.
    if live_query:
        logger.info(f"Attempting DuckDuckGo fallback for live query: {query}")
        for scoped_query in _build_live_domain_queries(query):
            result, has_scoped_live_signal = _search_duckduckgo(scoped_query, live_query=True)
            if result and has_scoped_live_signal:
                return f"{result}\n\n{_get_low_confidence_notice(language)}"
            logger.debug("Ignoring DuckDuckGo live result without recent signal: %s", scoped_query)
    else:
        logger.info(f"Falling back to DuckDuckGo for query: {query}")
        result, _ = _search_duckduckgo(search_query, live_query=False)
        if result:
            return f"{result}\n\n{_get_low_confidence_notice(language)}"

    # Final encyclopedia fallback for live-style queries when web search fails.
    if live_query:
        logger.info(f"No reliable live web source found for query: {query}")
        return _build_no_live_data_message(language=language, query=query)

    logger.info(f"Falling back to Wikipedia for query: {query}")
    result = _search_wikipedia_with_fallback(query, language)
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
        ("Castelo de São Jorge", "pt"),
        ("Greve Metro Lisboa", "pt"),
        ("What is happening with Lisbon metro service today?", "en"),
        ("History of Belém Tower", "en"),
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
