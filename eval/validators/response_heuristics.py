# ==========================================================================
# Master Thesis - Response Quality Heuristics
#   - André Filipe Gomes Silvestre, 20240502
#
# Deterministic, LLM-free heuristics for evaluating response quality.
# These complement the LLM-as-a-Judge by catching obvious issues that
# do not require semantic understanding (e.g., tool leaks, length,
# language compliance).
#
# Originally developed for the V1 evaluation framework.
# ==========================================================================

from __future__ import annotations

import re
from typing import Any

# -----------------------------------------------------------------------
# Individual Heuristic Functions
# -----------------------------------------------------------------------


def check_tool_leaks(response: str) -> dict[str, Any]:
    """Check if raw tool names or API artifacts leaked into user-facing text.

    Args:
        response: The generated response text.

    Returns:
        Dict with leaked (bool) and leaked_items (list of found leaks).
    """
    # Tool name patterns (snake_case function names)
    tool_patterns = [
        r"\bget_weather_\w+",
        r"\bget_metro_\w+",
        r"\bget_train_\w+",
        r"\bget_carris_\w+",
        r"\bcarris_get_\w+",
        r"\bfind_nearby_\w+",
        r"\bsearch_\w+_(?:events|attractions|knowledge)",
        r"\bget_route_between_stations\b",
        r"\bget_transport_summary\b",
        r"\blist_available_datasets\b",
        r"\bget_dataset_details\b",
        r"\bfind_place_in_datasets\b",
        r"\bfind_direct_bus_lines\b",
        r"\bget_real_time_bus_positions\b",
        r"\bplan_train_trip\b",
    ]
    # API artifact patterns
    artifact_patterns = [
        r"ToolMessage",
        r"tool_calls",
        r"AIMessage\(",
        r"HumanMessage\(",
        r"SystemMessage\(",
        r"\{\"tool_call_id\"",
        r"content=\[",
        r"additional_kwargs=",
    ]

    leaked_items = []
    for pattern in tool_patterns + artifact_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        leaked_items.extend(matches)

    return {"leaked": len(leaked_items) > 0, "leaked_items": leaked_items}


def check_response_length(
    response: str,
    min_chars: int = 20,
    max_chars: int = 5000,
    min_words: int = 5,
) -> dict[str, Any]:
    """Check if response length is within acceptable bounds.

    Args:
        response: The generated response text.
        min_chars: Minimum character count.
        max_chars: Maximum character count.
        min_words: Minimum word count.

    Returns:
        Dict with acceptable (bool), char_count, word_count, and any issues.
    """
    char_count = len(response.strip())
    word_count = len(response.split())
    issues = []

    if char_count < min_chars:
        issues.append(f"Too short: {char_count} chars (min: {min_chars})")
    if char_count > max_chars:
        issues.append(f"Too long: {char_count} chars (max: {max_chars})")
    if word_count < min_words:
        issues.append(f"Too few words: {word_count} (min: {min_words})")

    return {
        "acceptable": len(issues) == 0,
        "char_count": char_count,
        "word_count": word_count,
        "issues": issues,
    }


def check_language_compliance(
    response: str, expected_language: str
) -> dict[str, Any]:
    """Basic heuristic check for language compliance.

    Uses keyword detection to estimate if the response is in the expected
    language. This is a rough heuristic, not a full NLP language detector.

    Args:
        response: The generated response text.
        expected_language: ISO 639-1 code ("en", "pt", "fr").

    Returns:
        Dict with compliant (bool) and detected_indicators.
    """
    text_lower = response.lower()

    # Language indicator words (common function words)
    indicators = {
        "en": ["the", "is", "are", "you", "can", "this", "that", "with", "for", "from"],
        "pt": ["o", "a", "de", "do", "da", "em", "para", "com", "que", "uma"],
        "fr": ["le", "la", "les", "de", "des", "est", "sont", "vous", "pour", "avec"],
    }

    target_words = indicators.get(expected_language, [])
    if not target_words:
        return {"compliant": True, "detected_indicators": {}, "note": "No indicators for language"}

    # Count indicator words
    word_set = set(re.findall(r"\b\w+\b", text_lower))
    detected = {}
    for lang, words in indicators.items():
        count = sum(1 for w in words if w in word_set)
        detected[lang] = count

    # The expected language should have the highest count (or tied)
    max_count = max(detected.values()) if detected else 0
    expected_count = detected.get(expected_language, 0)
    compliant = expected_count >= max_count and expected_count > 0

    return {
        "compliant": compliant,
        "detected_indicators": detected,
        "expected_language": expected_language,
    }


def check_hallucinated_features(response: str) -> dict[str, Any]:
    """Check if the response claims capabilities the system does not have.

    Args:
        response: The generated response text.

    Returns:
        Dict with hallucinated (bool) and flagged_claims (list).
    """
    text_lower = response.lower()
    unsupported_claims = [
        (r"\bbook(?:ing|ed)?\b.*\b(?:table|hotel|flight|ticket|restaurant)\b",
         "Booking capability"),
        (r"\b(?:purchase|buy|order)\b",
         "Purchase/order capability"),
        (r"\breal[\s-]?time\s+(?:traffic|congestion)\b",
         "Real-time traffic data"),
        (r"\bfertagus\b.*\bschedule\b",
         "Fertagus schedule data"),
        (r"\b(?:uber|bolt|lyft|taxi)\b.*\b(?:price|cost|fare)\b",
         "Ride-hailing pricing"),
        (r"\bhistorical\s+(?:weather|climate)\s+(?:data|average|record)\b",
         "Historical climate data"),
    ]

    flagged = []
    for pattern, description in unsupported_claims:
        if re.search(pattern, text_lower):
            flagged.append(description)

    return {"hallucinated": len(flagged) > 0, "flagged_claims": flagged}


def check_emoji_density(response: str, max_ratio: float = 0.05) -> dict[str, Any]:
    """Check if emoji usage is excessive.

    Args:
        response: The generated response text.
        max_ratio: Maximum ratio of emoji characters to total characters.

    Returns:
        Dict with acceptable (bool), emoji_count, and ratio.
    """
    # Match emoji characters (common ranges)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    emojis = emoji_pattern.findall(response)
    emoji_count = sum(len(e) for e in emojis)
    total = max(len(response), 1)
    ratio = emoji_count / total

    return {
        "acceptable": ratio <= max_ratio,
        "emoji_count": emoji_count,
        "ratio": round(ratio, 4),
    }


# -----------------------------------------------------------------------
# Aggregate Function
# -----------------------------------------------------------------------

def run_all_heuristics(
    response: str,
    expected_language: str = "en",
) -> dict[str, Any]:
    """Run all response quality heuristics and return aggregate results.

    Args:
        response: The generated response text.
        expected_language: Expected response language (ISO 639-1).

    Returns:
        Dict with individual check results and overall pass/fail.
    """
    results = {
        "tool_leaks": check_tool_leaks(response),
        "response_length": check_response_length(response),
        "language_compliance": check_language_compliance(response, expected_language),
        "hallucinated_features": check_hallucinated_features(response),
        "emoji_density": check_emoji_density(response),
    }

    # Overall: pass if no critical failures
    critical_failures = []
    if results["tool_leaks"]["leaked"]:
        critical_failures.append("tool_leaks")
    if not results["response_length"]["acceptable"]:
        critical_failures.append("response_length")
    if results["hallucinated_features"]["hallucinated"]:
        critical_failures.append("hallucinated_features")

    results["overall_pass"] = len(critical_failures) == 0
    results["critical_failures"] = critical_failures

    return results


# -----------------------------------------------------------------------
# Self-Tests
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("\033[1m=== Response Heuristics Self-Tests ===\033[0m\n")

    # Test 1: Clean response
    clean = "The weather in Lisbon today is sunny with temperatures around 22 degrees Celsius."
    r = run_all_heuristics(clean, "en")
    assert r["overall_pass"], f"Clean response should pass: {r['critical_failures']}"
    print("\033[1;32m[PASS]\033[0m Clean response passes all checks")

    # Test 2: Tool leak detection
    leaked = "Based on get_weather_forecast, the temperature will be 25C."
    r = check_tool_leaks(leaked)
    assert r["leaked"], "Should detect tool name leak"
    print("\033[1;32m[PASS]\033[0m Tool leak detected: get_weather_forecast")

    # Test 3: API artifact leak
    artifact = 'The result is ToolMessage(content="sunny").'
    r = check_tool_leaks(artifact)
    assert r["leaked"], "Should detect ToolMessage leak"
    print("\033[1;32m[PASS]\033[0m API artifact detected: ToolMessage")

    # Test 4: Too short response
    short = "OK."
    r = check_response_length(short)
    assert not r["acceptable"], "2-char response should fail"
    print("\033[1;32m[PASS]\033[0m Too short response detected")

    # Test 5: Language compliance (PT response for PT query)
    pt_resp = "O tempo em Lisboa hoje é ensolarado com temperaturas de 22 graus."
    r = check_language_compliance(pt_resp, "pt")
    assert r["compliant"], f"Portuguese response should be compliant: {r}"
    print("\033[1;32m[PASS]\033[0m Portuguese language compliance")

    # Test 6: Language mismatch (EN response for PT query)
    en_resp = "The weather in Lisbon today is sunny with temperatures around 22 degrees."
    r = check_language_compliance(en_resp, "pt")
    assert not r["compliant"], "English response should not match PT expectation"
    print("\033[1;32m[PASS]\033[0m Language mismatch detected (EN vs PT)")

    # Test 7: Hallucinated feature detection
    halluc = "I can book you a table at the restaurant for tomorrow evening."
    r = check_hallucinated_features(halluc)
    assert r["hallucinated"], "Booking claim should be flagged"
    print("\033[1;32m[PASS]\033[0m Hallucinated booking capability detected")

    # Test 8: Emoji density check
    emoji_heavy = "🌞🌤️☀️ The weather is great! 🎉🎊🥳 Visit the museums! 🏛️🖼️🎨"
    r = check_emoji_density(emoji_heavy, max_ratio=0.05)
    assert not r["acceptable"], "Heavy emoji usage should fail"
    print("\033[1;32m[PASS]\033[0m Excessive emoji density detected")

    # Test 9: No hallucination for valid response
    valid = "The Lisbon Metro has four lines. I recommend taking the Green line."
    r = check_hallucinated_features(valid)
    assert not r["hallucinated"], "Valid response should not flag hallucinations"
    print("\033[1;32m[PASS]\033[0m No false hallucination flag")

    # Test 10: Aggregate with failures
    bad = "get_metro_status OK."
    r = run_all_heuristics(bad, "en")
    assert not r["overall_pass"], "Should fail with tool leak + too short"
    assert "tool_leaks" in r["critical_failures"]
    print("\033[1;32m[PASS]\033[0m Aggregate failure correctly detected")

    print("\n\033[1;32m=== All 10 tests passed ===\033[0m")
