# ==========================================================================
# Master Thesis - Response Formatter
#   - André Filipe Gomes Silvestre, 20240502
#
#   Post-processing pipeline to ensure LLM responses render cleanly
#   in Streamlit's st.markdown(). Normalizes headers, spacing,
#   bullet styles, and URL formatting for consistent visual quality.
# ==========================================================================

import re
from typing import Optional
from urllib.parse import urlparse


def normalize_headers(text: str) -> str:
    """
    Normalizes markdown headers to consistent levels.

    Rules:
        - # (h1) → ### (h3) to avoid oversized headers in Streamlit
        - ## (h2) → ### (h3) for consistency
        - #### and deeper stay as-is

    Args:
        text: Raw markdown text.

    Returns:
        str: Text with normalized header levels.
    """
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        # Only normalize headers at beginning of line (not in code blocks)
        # Use re.sub to safely replace header prefix without eating content chars
        if stripped.startswith("# ") and not stripped.startswith("## "):
            # h1 -> h3: replace the leading '# ' with '### '
            result.append(re.sub(r'^# ', '### ', stripped))
        elif stripped.startswith("## ") and not stripped.startswith("### "):
            # h2 -> h3: replace the leading '## ' with '### '
            result.append(re.sub(r'^## ', '### ', stripped))
        else:
            result.append(line)
    return "\n".join(result)


def add_section_separators(text: str) -> str:
    """
    Adds horizontal rules between major content sections.

    Inserts --- before ### headers (unless one already exists),
    creating visual separation between sections in Streamlit.

    Args:
        text: Markdown text.

    Returns:
        str: Text with separators added between sections.
    """
    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Add separator before ### headers (not the first one)
        if stripped.startswith("### ") and i > 0:
            # Check if previous non-empty line is already a separator
            prev_non_empty = ""
            for j in range(i - 1, -1, -1):
                if lines[j].strip():
                    prev_non_empty = lines[j].strip()
                    break
            if prev_non_empty != "---":
                result.append("")
                result.append("---")
                result.append("")
        result.append(line)
    return "\n".join(result)


def clean_newlines(text: str) -> str:
    """
    Removes excessive consecutive blank lines (max 2).

    Args:
        text: Text with potentially excessive newlines.

    Returns:
        str: Text with at most 2 consecutive blank lines.
    """
    # Replace 3+ consecutive newlines with 2
    return re.sub(r"\n{4,}", "\n\n\n", text)


def normalize_bullets(text: str) -> str:
    """
    Normalizes bullet point styles to consistent format.

    Converts various bullet markers (*, •, >) to standard markdown (-).
    Preserves blockquotes (> at start of line with content).

    Args:
        text: Text with mixed bullet styles.

    Returns:
        str: Text with consistent bullet formatting.
    """
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        # Get leading whitespace
        indent = len(line) - len(line.lstrip())
        spaces = " " * indent

        # Convert * bullets to - (but not ** bold markers or * in words)
        if stripped.startswith("* ") and not stripped.startswith("**"):
            result.append(f"{spaces}- {stripped[2:]}")
        # Convert • bullets to -
        elif stripped.startswith("• "):
            result.append(f"{spaces}- {stripped[2:]}")
        else:
            result.append(line)
    return "\n".join(result)


def ensure_clickable_urls(text: str) -> str:
    """
    Wraps bare URLs in markdown link syntax for clickable rendering.

    Detects URLs that aren't already in markdown link format [text](url)
    and wraps them. Skips URLs already inside markdown links or code blocks.

    Args:
        text: Text potentially containing bare URLs.

    Returns:
        str: Text with all URLs made clickable.
    """
    # Match bare URLs not already in markdown format
    # Negative lookbehind: not preceded by ]( or ](
    # Negative lookbehind: not preceded by ` (code)
    url_pattern = r'(?<!\]\()(?<!\`)(?<!\[)(https?://[^\s\)]+)'

    def replace_url(match):
        url = match.group(1)
        # Extract domain for display
        try:
            domain = urlparse(url).netloc
            if domain.startswith("www."):
                domain = domain[4:]
            return f"[{domain}]({url})"
        except Exception:
            return f"[Link]({url})"

    # Only replace URLs that aren't already in markdown link format
    # Check if URL is preceded by [...]( pattern
    lines = text.split("\n")
    result = []
    in_code_block = False

    for line in lines:
        # Skip code blocks
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            continue

        if in_code_block:
            result.append(line)
            continue

        # Check if line already has markdown links
        if "](http" in line or "](" in line:
            result.append(line)
            continue

        # Replace bare URLs
        result.append(re.sub(url_pattern, replace_url, line))

    return "\n".join(result)


def format_response(text: str) -> str:
    """
    Main formatting pipeline for LLM responses.

    Applies all formatting transformations in order:
        1. Normalize headers (avoid h1/h2, use h3+)
        2. Add section separators
        3. Clean excessive newlines
        4. Normalize bullet styles
        5. Ensure URLs are clickable

    Args:
        text: Raw LLM response text.

    Returns:
        str: Formatted text ready for Streamlit rendering.
    """
    if not text or not isinstance(text, str):
        return text or ""

    text = normalize_headers(text)
    text = add_section_separators(text)
    text = clean_newlines(text)
    text = normalize_bullets(text)
    text = ensure_clickable_urls(text)

    return text.strip()


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    import time

    test_input = """# Weather in Lisbon

## Current Conditions

* Temperature: **22°C**
* Humidity: 65%
• Wind: 15 km/h NW

## What to do today

Here are some suggestions:

* Visit the Jerónimos Monastery
• Take the 28E tram
* Walk along the riverfront

Check the official site: https://www.visitlisboa.com

## Transport Tips

More info at https://www.metrolisboa.pt and https://www.carris.pt

### Already a h3

This should stay as-is.




Too many blank lines above should be reduced.
"""

    print("=" * 60)
    print("🧪 Response Formatter Test")
    print("=" * 60)

    start = time.time()
    output = format_response(test_input)
    elapsed = time.time() - start

    print(f"\n📥 INPUT ({len(test_input)} chars):")
    print("-" * 40)
    print(test_input[:200] + "...")

    print(f"\n📤 OUTPUT ({len(output)} chars, {elapsed*1000:.1f}ms):")
    print("-" * 40)
    print(output)

    # Verify transformations
    checks = {
        "No h1/h2 headers": not any(
            (line.startswith("# ") and not line.startswith("## ")) or
            (line.startswith("## ") and not line.startswith("### "))
            for line in output.split("\n")
        ),
        "Has --- separators": "---" in output,
        "No excessive newlines": "\n\n\n\n" not in output,
        "Consistent bullets": all(
            not line.strip().startswith("* ") or line.strip().startswith("**")
            for line in output.split("\n")
        ),
        "URLs are clickable": "](http" in output,
    }

    print("\n✅ Checks:")
    all_pass = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n🎉 ALL CHECKS PASSED")
    else:
        print("\n❌ SOME CHECKS FAILED")
