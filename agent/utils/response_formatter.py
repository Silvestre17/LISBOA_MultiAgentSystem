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
    # First: convert setext-style headers (underline with === or ---) to ATX style
    # "Title\n=====" -> "# Title"  and  "Title\n-----" -> "## Title"
    text = re.sub(r'^(.+)\n={3,}\s*$', r'### \1', text, flags=re.MULTILINE)
    text = re.sub(r'^(.+)\n-{3,}\s*$', r'### \1', text, flags=re.MULTILINE)

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
    Removes excessive consecutive blank lines (max 1 blank line between content).

    Args:
        text: Text with potentially excessive newlines.

    Returns:
        str: Text with at most 2 consecutive newlines (1 blank line).
    """
    # Replace 3+ consecutive newlines with 2 (single blank line)
    return re.sub(r"\n{3,}", "\n\n", text)


def normalize_bullets(text: str) -> str:
    """
    Normalizes bullet point styles to consistent format, ensures labels are bold,
    and adds tight spacing using markdown hard breaks.

    Rules:
    - Lists with emojis do not get standard bullets, they use the emoji.
    - Numbered lists are bolded automatically (e.g., '1.' -> '**1.**')
    - Labels (e.g., 'Morada:', 'Preço:') are bolded automatically.
    - Two spaces are appended to lists and sub-items for tight <br> spacing.
    - Removes dummy TripAdvisor '⭐ 4.5/5' appended to all events
    - Suppresses repeated '⚠️ Nota:' remarks from IPMA.

    Args:
        text: Text to format.

    Returns:
        str: Formatted text.
    """
    lines = text.split("\n")
    out = []
    
    emoji_pattern = re.compile(r'^[\u2600-\U0010ffff\u2B50\u200D\uFE0F]{1,3}')
    # Match labels (e.g. "Data/Hora:", "Preço: ") optionally prefixed by emoji
    label_pattern = re.compile(r'^([\u2600-\U0010ffff\u2B50\u200D\uFE0F]{1,3}\s*)?([A-Za-zÀ-ÿ/\s]{3,25}):\s*(.*)')
    
    # Matches the useless ratings added by VisitLisboa tool
    remove_stars_pattern = re.compile(r'\s*-\s*⭐\s*4\.5/5\s*$')
    # Filter repeated 'Nota:' elements
    filter_nota_pattern = re.compile(r'^(?:⚠️\s*)?Nota:', re.IGNORECASE)
    
    nota_count = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
            
        m_nota = filter_nota_pattern.match(stripped)
        if m_nota:
            nota_count += 1
            if nota_count > 1:
                continue
                
        # Remove dummy stars
        stripped = remove_stars_pattern.sub('', stripped)
            
        indent = len(line) - len(line.lstrip())
        spaces = " " * indent
        
        # Determine if it's a bulleted line
        is_bullet = stripped.startswith("- ") or stripped.startswith("* ") or stripped.startswith("• ")
        
        if is_bullet:
            content = stripped[2:].strip()
        else:
            content = stripped
            
        # Detect numbered lists and format labels (Data/Hora: -> **Data/Hora**: )
        if "**" not in content and not content.startswith("#"):
            m_num = re.match(r'^(\d+\.)\s+(.*)', content)
            if m_num:
                num = m_num.group(1)
                rest = m_num.group(2)
                content = f"**{num}** {rest}"
            else:
                m_label = label_pattern.match(content)
                if m_label:
                    emoji_part = m_label.group(1) or ""
                    label = m_label.group(2).strip()
                    rest = m_label.group(3)
                    content = f"{emoji_part}**{label}**: {rest}"
                
        # Format the output block
        if is_bullet:
            clean_content = content.replace('**', '').strip()
            # If the content starts with an emoji, variant selector, or is a numbered item (**1.**)
            is_numbered = re.match(r'^\*\*\d+\.\*\*', content)
            
            if emoji_pattern.match(clean_content) or clean_content.startswith('️') or is_numbered:
                out.append(f"{spaces}- {content}")
            else:
                out.append(f"{spaces}- {content}")
        else:
            # Non-bullet line: use modified content (with auto-bolded labels/numbers)
            # if content was changed, rebuild the line preserving indentation
            if content != stripped:
                out.append(f"{spaces}{content}")
            else:
                out.append(line)
                
    return "\n".join(out)


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
    # Match bare URLs not preceded by ]( (already a markdown link target)
    url_pattern = re.compile(r'(?<!\]\()(https?://[^\s\)\]]+)')
    # Pattern for existing markdown links: [text](url)
    md_link_pattern = re.compile(r'\[[^\]]*\]\([^)]+\)')

    def replace_url(match):
        url = match.group(1)
        try:
            domain = urlparse(url).netloc
            if domain.startswith("www."):
                domain = domain[4:]
            return f"[{domain}]({url})"
        except Exception:
            return f"[Link]({url})"

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

        # Find existing markdown links in the line and protect their URLs
        # Replace bare URLs only outside of markdown link constructs
        existing_links = [(m.start(), m.end()) for m in md_link_pattern.finditer(line)]

        if not existing_links:
            # No existing links - safe to replace all bare URLs
            result.append(url_pattern.sub(replace_url, line))
        else:
            # Has existing links - only replace URLs outside of them
            new_line = []
            last_end = 0
            for start, end in existing_links:
                # Process text before the markdown link (may have bare URLs)
                segment = line[last_end:start]
                new_line.append(url_pattern.sub(replace_url, segment))
                # Keep the markdown link as-is
                new_line.append(line[start:end])
                last_end = end
            # Process remaining text after last markdown link
            segment = line[last_end:]
            new_line.append(url_pattern.sub(replace_url, segment))
            result.append("".join(new_line))

    return "\n".join(result)


def strip_internal_sections(text: str) -> str:
    """
    Removes sections that expose internal system details (QA, disclaimers, etc.)
    that should never appear in user-facing responses.

    Matches header-based sections like:
        - ### Observações e disclaimers
        - ### Checklist de Completude
        - ### Quality Check
        - ### QA Results / Data Validation
        - ### Fonte & Observações (when it contains QA content)

    Args:
        text: Formatted markdown text.

    Returns:
        str: Text with internal sections removed.
    """
    # Patterns for internal section headers (case-insensitive)
    internal_patterns = [
        r'observa[çc][õo]es\s+e\s+disclaimers',
        r'checklist\s+de\s+completude',
        r'quality\s+(check|assurance)',
        r'qa[\s_]+(results?|validation|disclaimers?)',
        r'data\s+validation',
        r'completeness\s+check',
        r'disclaimers?\s*$',
        r'notas?\s+de\s+qualidade',
        r'controlo\s+de\s+qualidade',
    ]
    combined = '|'.join(internal_patterns)
    # Match headers (any level) containing these patterns
    header_re = re.compile(
        r'^(#{1,6})\s+.*?(' + combined + r').*$',
        re.IGNORECASE | re.MULTILINE
    )

    lines = text.split('\n')
    result = []
    skip_until_next_header = False
    skip_header_level = 0

    for line in lines:
        stripped = line.strip()

        # Check if this is a header
        header_match = re.match(r'^(#{1,6})\s+', stripped)

        if header_match:
            level = len(header_match.group(1))

            if skip_until_next_header:
                # Found a new header - check if same or higher level (stop skipping)
                if level <= skip_header_level:
                    skip_until_next_header = False
                else:
                    # Sub-header of the skipped section, continue skipping
                    continue

            # Check if this header matches an internal pattern
            if header_re.match(stripped):
                skip_until_next_header = True
                skip_header_level = level
                continue

        if skip_until_next_header:
            continue

        result.append(line)

    return '\n'.join(result)


def add_section_spacing(text: str) -> str:
    """
    Ensures blank lines before section-like markers so content blocks
    are visually separated (e.g. Avisos, Dicas, Fonte, Nota, headers).

    This prevents different content blocks from appearing 'cramped'
    together without any breathing room.

    Args:
        text: Markdown text to process.

    Returns:
        str: Text with proper spacing before section markers.
    """
    # Patterns that should always have a blank line before them
    section_markers = [
        r"^#{1,4}\s",                     # Any markdown header
        r"^(?:⚠️|⚠)\s*\*\*(?:Avisos|Aviso|Warnings?|Nota|Note)",
        r"^💡\s*\*\*(?:Dicas?|Tips?|Sugest)",
        r"^📌\s*\*\*(?:Fonte|Source)",
        r"^🌡️",                           # Weather emoji section
        r"^🌤️",
        r"^🌧️",
        r"^---\s*$",                       # Horizontal rules
    ]
    combined = "|".join(f"(?:{p})" for p in section_markers)
    section_re = re.compile(combined)

    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i > 0 and section_re.match(stripped):
            # Check if previous line is already blank
            prev = result[-1].strip() if result else ""
            if prev != "":
                result.append("")
        result.append(line)

    return "\n".join(result)


def format_response(text: str) -> str:
    """
    Main formatting pipeline for LLM responses.

    Applies all formatting transformations in order:
        1. Strip internal/QA sections that should never reach the user
        2. Normalize headers (avoid h1/h2, use h3+)
        3. Clean excessive newlines
        4. Add spacing between distinct content sections
        5. Normalize bullet styles
        6. Ensure URLs are clickable

    Note:
        Section separators (---) are NOT injected automatically.
        They create excessive visual spacing in Streamlit. Agents can
        include them explicitly when needed.

    Args:
        text: Raw LLM response text.

    Returns:
        str: Formatted text ready for Streamlit rendering.
    """
    if not text or not isinstance(text, str):
        return text or ""

    text = strip_internal_sections(text)
    text = normalize_headers(text)
    text = clean_newlines(text)
    text = add_section_spacing(text)
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

* 🌡️ Temperature: **22°C**
* 💧 Humidity: 65%
• 🌬️ Wind: 15 km/h NW
* Normal bullet without emoji

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
        "No excessive newlines": "\n\n\n" not in output,
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
