# ==========================================================================
# Master Thesis - Response Formatter
#   - André Filipe Gomes Silvestre, 20240502
#
#   Post-processing pipeline to ensure LLM responses render cleanly
#   in Streamlit's st.markdown(). Includes link normalization,
#   metro terminology cleanup, header and bullet normalization,
#   response-title helpers, and final formatting for consistent visual quality.
# ==========================================================================

import re
from typing import Optional
from urllib.parse import urlparse


def normalize_source_links(text: str) -> str:
    """
    Normalizes malformed HTML anchor tags and bare Metro de Lisboa source text
    into standard markdown links that Streamlit renders correctly.

    Args:
        text: Raw LLM response text.

    Returns:
        str: Text with standardized Metro de Lisboa source links.
    """
    if not text:
        return text

    # Convert malformed/HTML anchors for Metro de Lisboa to markdown.
    text = re.sub(
        r'<a\s+href="?(?:https?://)?metrolisboa\.pt"?[^>]*>\s*Metro de Lisboa\s*</a>',
        r'[*Metro de Lisboa*](https://www.metrolisboa.pt)',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'<a\s+href="?https?://www\.metrolisboa\.pt"?[^>]*>\s*Metro de Lisboa\s*</a>',
        r'[*Metro de Lisboa*](https://www.metrolisboa.pt)',
        text,
        flags=re.IGNORECASE,
    )
    return text


def normalize_metro_terminology(text: str) -> str:
    """
    Fixes incorrect rail terminology when the response is clearly about
    Metro de Lisboa routes.

    Args:
        text: Raw LLM response text.

    Returns:
        str: Text with metro terminology normalized.
    """
    if not text:
        return text

    metro_context = re.search(
        r'(O seu Trajeto de Metro|Próximos Metros|Metro de Lisboa|Linha Azul|Linha Verde|Linha Amarela|Linha Vermelha)',
        text,
        re.IGNORECASE,
    )
    cp_context = re.search(r'\bCP\b|Comboios de Portugal|CP Trains', text, re.IGNORECASE)

    if metro_context and not cp_context:
        replacements = [
            (r'\bcomboios\b', 'metros'),
            (r'\bComboios\b', 'Metros'),
            (r'\bcomboio\b', 'metro'),
            (r'\bComboio\b', 'Metro'),
            (r'\btrems\b', 'metros'),
            (r'\bTrems\b', 'Metros'),
            (r'\btrem\b', 'metro'),
            (r'\bTrem\b', 'Metro'),
            (r'transferência provável', 'transferência'),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)

    return text


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
            # Normalize all bullet variants to standard Markdown "- "
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


def clean_decorative_separators(text: str) -> str:
    """
    Removes decorative separator lines that are not valid markdown.

    Lines consisting only of repeated '=' or '-' characters (5 or more)
    are removed. Valid markdown horizontal rules ('---' exactly 3 dashes
    on a line) are preserved.

    This acts as a safety net for tool outputs that use decorative
    separators like '=' * 50 or '-' * 30 which render as plain text
    in Streamlit rather than as visual dividers.

    Args:
        text: Text potentially containing decorative separators.

    Returns:
        str: Text with decorative separators removed.
    """
    # Remove lines of 5+ repeated = or - characters (but preserve --- which is valid markdown)
    # Also preserve lines with mixed content (e.g., '--- some text')
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are ONLY repeated = or - (5+ chars), these are decorative
        if len(stripped) >= 5 and all(c in '=-' for c in stripped):
            continue
        result.append(line)
    return "\n".join(result)


def generate_response_title(
    agents_called: list,
    user_query: str,
    language: str = "en",
) -> Optional[str]:
    """
    Generates a contextual ### (h3) title for the response based on routing.

    Returns None for direct responses (no agents), planner responses,
    or greetings, so they remain untitled.

    Args:
        agents_called: List of agent names invoked (e.g., ["weather"]).
        user_query: The original user query.
        language: Language code ('en' or 'pt').

    Returns:
        Optional[str]: A markdown ### title string, or None.
    """
    if not agents_called:
        return None  # Direct response (OOS, greeting) - no title

    if "planner" in agents_called:
        return None  # Planner generates its own header

    query_lower = user_query.lower()

    # --- Single agent responses ---
    if len(agents_called) == 1:
        agent = agents_called[0]

        if agent == "weather":
            return (
                "### \U0001f324\ufe0f Previsão Meteorológica"
                if language == "pt"
                else "### \U0001f324\ufe0f Weather Forecast"
            )

        elif agent == "transport":
            return (
                "### \U0001f687 Informação de Transportes"
                if language == "pt"
                else "### \U0001f687 Transport Information"
            )

        elif agent == "researcher":
            # Keyword-based subcategorization
            event_kw = [
                "evento", "event", "concerto", "concert", "festival",
                "espetáculo", "show", "teatro", "theatre", "theater",
                "ópera", "opera", "dança", "dance", "exposição", "exhibition",
            ]
            place_kw = [
                "museu", "museum", "monumento", "monument", "castelo", "castle",
                "igreja", "church", "torre", "tower", "praça", "square",
                "bairro", "neighborhood", "miradouro", "viewpoint", "jardim",
                "garden", "parque", "park",
            ]
            food_kw = [
                "restaurante", "restaurant", "comida", "food", "comer", "eat",
                "café", "coffee", "bar", "pastelaria", "bakery",
                "gastronomia", "gastronomy", "nightlife", "vida noturna",
            ]
            service_kw = [
                "farmácia", "pharmacy", "hospital", "escola", "school",
                "biblioteca", "library", "polícia", "police", "bombeiros",
                "fire", "wc", "sanitário", "mercado", "market", "creche",
                "estacionamento", "parking", "feira", "marketplace",
            ]
            history_kw = [
                "história", "history", "cultura", "culture", "origem", "origin",
                "fundação", "founded", "tradição", "tradition",
            ]

            if any(kw in query_lower for kw in event_kw):
                return (
                    "### \U0001f3ad Eventos Culturais"
                    if language == "pt"
                    else "### \U0001f3ad Cultural Events"
                )
            elif any(kw in query_lower for kw in place_kw):
                return (
                    "### \U0001f4cd Locais e Atrações"
                    if language == "pt"
                    else "### \U0001f4cd Places & Attractions"
                )
            elif any(kw in query_lower for kw in food_kw):
                return (
                    "### \U0001f37d\ufe0f Gastronomia"
                    if language == "pt"
                    else "### \U0001f37d\ufe0f Food & Dining"
                )
            elif any(kw in query_lower for kw in service_kw):
                return (
                    "### \U0001f3e5 Serviços Essenciais"
                    if language == "pt"
                    else "### \U0001f3e5 Essential Services"
                )
            elif any(kw in query_lower for kw in history_kw):
                return (
                    "### \U0001f4da História e Cultura"
                    if language == "pt"
                    else "### \U0001f4da History & Culture"
                )
            else:
                return (
                    "### \U0001f4cb Informação Local"
                    if language == "pt"
                    else "### \U0001f4cb Local Information"
                )

    # --- Multi-agent (without planner) - combined titles ---
    if "weather" in agents_called and "transport" in agents_called:
        return (
            "### \U0001f324\ufe0f\U0001f687 Meteorologia e Transportes"
            if language == "pt"
            else "### \U0001f324\ufe0f\U0001f687 Weather & Transport"
        )
    elif "weather" in agents_called:
        return (
            "### \U0001f324\ufe0f Previsão Meteorológica"
            if language == "pt"
            else "### \U0001f324\ufe0f Weather Forecast"
        )
    elif "transport" in agents_called:
        return (
            "### \U0001f687 Informação de Transportes"
            if language == "pt"
            else "### \U0001f687 Transport Information"
        )
    else:
        return (
            "### \U0001f4cb Informação Local"
            if language == "pt"
            else "### \U0001f4cb Local Information"
        )


def ensure_response_title(text: str, title: Optional[str]) -> str:
    """
    Prepends a contextual title to the response if it doesn't already have one.

    Skips prepending if:
        - title is None or empty
        - response already starts with a markdown header (###, ##, #)
        - response already starts with a bold title (**Title**)

    Args:
        text: Formatted response text.
        title: The ### title to prepend, or None.

    Returns:
        str: Response with title prepended (or unchanged).
    """
    if not title or not text:
        return text or ""

    # Check if response already starts with a header or bold title
    first_line = text.strip().split("\n")[0].strip()
    if first_line.startswith("### ") or first_line.startswith("## ") or first_line.startswith("# "):
        return text  # Already has a header
    if re.match(r"^\*\*[^*]+\*\*\s*$", first_line):
        return text  # Already has a bold title line

    return f"{title}\n\n{text}"



def strip_hallucinations(text: str) -> str:
    if not text:
        return ""

    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        if re.match(r"^(?:\s*|-\s*|\*\s*|\**|\[|\]|\*|#|>)*\s*(Introdu[cç][aã]o|Introduction)\b", line, re.IGNORECASE):
            continue
        if re.match(r"^(?:\s*|-\s*|\*\s*|\**|\[|\]|\*|#|>)*\s*(Contrainte do utilizador|Restri[cç][õo]es do utilizador|How the response meets|Acessibilidade/Tempo/Budget|Accessibility/Time/Budget)\b", line, re.IGNORECASE):
            continue
        if re.match(r"^(?:\s*|-\s*|\*\s*|\**|\[|\]|\*|#|>)*\s*(Observa[cç][aã]o|Observa[cç][õo]es|Nota|Notes?):?", line, re.IGNORECASE):
            continue
        if re.match(r"^(?:\s*|-\s*|\*\s*|\**|\[|\]|\*|#|>)*\s*(Diga se|Se quiser|Se quiseres|Se preferir|Posso ajudar|Posso detalhar|I can also|I can help|Let me know):?", line, re.IGNORECASE):
            continue
        if re.match(r"^(?:\s*|-\s*|\*\s*|\**|\[|\]|\*|#|>)*\s*(⭐\s*Rating:\s*(Sem avaliação de rating|No rating available))\s*$", line, re.IGNORECASE):
            continue
        if "Não listado o Opposto" in line or "opposite direction" in line.lower():
            continue
        clean_lines.append(line)
    text = "\n".join(clean_lines)

    # Normalize source emphasis before truncating.
    text = re.sub(r"Fonte:\s*📌\s*Fonte:\s*", "📌 **Fonte:** ", text, flags=re.IGNORECASE)
    text = re.sub(r"^Fonte:\s*", "📌 **Fonte:** ", text, flags=re.MULTILINE)
    text = re.sub(r"📌\s*Fonte:", "📌 **Fonte:**", text)
    text = re.sub(r"\|\s*Atualizado:", "| **Atualizado:**", text)
    text = re.sub(r"\|\s*Updated:", "| **Updated:**", text)
    text = re.sub(r"\*\*\|\s*\*\*(Atualizado|Updated):\*+", r"| **\1:**", text)
    text = text.replace("**| **Atualizado:****", "| **Atualizado:**")
    text = text.replace("**| **Updated:****", "| **Updated:**")

    # Hard truncate after the first valid source line.
    match = re.search(r"^(📌\s*\*\*Fonte:\*\*.*?(?:Atualizado|Updated):\s*\d{2}:\d{2}).*$", text, re.MULTILINE)
    if match:
        text = (text[:match.start()] + match.group(1)).rstrip()
    else:
        match2 = re.search(r"^(📌\s*Fonte:.*?(?:Atualizado|Updated):\s*\d{2}:\d{2}).*$", text, re.MULTILINE)
        if match2:
            text = (text[:match2.start()] + match2.group(1)).rstrip()
        else:
            match3 = re.search(r"^(📌\s*\*\*Fonte:\*\*.*)$", text, re.MULTILINE)
            if match3:
                text = (text[:match3.start()] + match3.group(1)).rstrip()

    return text


def format_response(text: str) -> str:
    """
    Main formatting pipeline for LLM responses.

    Applies all formatting transformations in order:
        1. Strip internal/QA sections that should never reach the user
        2. Clean decorative separators (e.g., '=' * 50, '-' * 30)
        3. Normalize headers (avoid h1/h2, use h3+)
        4. Add spacing between distinct content sections
        5. Add section separators (---) before ### headers
        6. Clean excessive newlines (after all spacing steps)
        7. Normalize bullet styles
        8. Ensure URLs are clickable

    Args:
        text: Raw LLM response text.

    Returns:
        str: Formatted text ready for Streamlit rendering.
    """
    if not text or not isinstance(text, str):
        return text or ""

    text = normalize_source_links(text)
    text = normalize_metro_terminology(text)
    text = strip_hallucinations(text)
    text = strip_internal_sections(text)
    text = clean_decorative_separators(text)
    text = normalize_headers(text)
    text = add_section_spacing(text)
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
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check}")
        if not result:
            all_pass = False

    # --- generate_response_title() tests ---
    print("\n\033[1m🔤 generate_response_title() Tests:\033[0m")
    # Signature: (agents_called: list, user_query: str, language: str) -> Optional[str]
    title_cases = [
        (["weather"], "weather forecast lisbon", "en", "### "),
        (["weather"], "tempo em lisboa amanhã", "pt", "### "),
        (["transport"], "próximo metro rossio", "pt", "### "),
        (["transport"], "bus schedule to Cascais", "en", "### "),
        (["researcher"], "exposição no museu", "pt", "### "),
        (["researcher"], "museum near alfama", "en", "### "),
        (["researcher"], "jantar no bairro alto", "pt", "### "),
        (["researcher"], "restaurant recommendations", "en", "### "),
        (["planner"], "plan my full day in lisbon", "en", None),
        ([], "olá bom dia", "pt", None),
    ]
    title_pass = 0
    for agents, query, lang, expected in title_cases:
        title = generate_response_title(agents, query, language=lang)
        if expected is None:
            ok = title is None
        else:
            ok = title is not None and title.startswith(expected)
        status = "✅" if ok else "❌"
        print(f"  {status} [{lang}] agents={agents} '{query}' → {title!r}")
        if ok:
            title_pass += 1
        else:
            all_pass = False
    print(f"  → {title_pass}/{len(title_cases)} title tests passed")

    # --- ensure_response_title() tests ---
    print("\n\033[1m📌 ensure_response_title() Tests:\033[0m")
    # Signature: (text: str, title: Optional[str]) -> str
    ensure_cases = [
        ("Some content without a header.", "### 🌤️ Weather in Lisbon", True),
        ("### Existing Header\nContent", "### 🚇 Transport", False),
        ("**Bold Title**\nContent", "### 🎭 Events", False),
        ("Some content", None, False),
        ("", "### 🎭 Events", False),
    ]
    ensure_pass = 0
    for text_in, title_in, expect_injected in ensure_cases:
        result = ensure_response_title(text_in, title_in)
        if expect_injected:
            ok = result.lstrip().startswith("### ") and str(title_in) in result
        elif text_in == "":
            ok = result == ""
        elif title_in is None:
            ok = result == text_in
        else:
            ok = result == text_in
        status = "✅" if ok else "❌"
        label = "(injected)" if expect_injected else "(unchanged)"
        print(f"  {status} {label}: title={str(title_in)[:25]!r} → {result[:50]!r}...")
        if ok:
            ensure_pass += 1
        else:
            all_pass = False
    print(f"  → {ensure_pass}/{len(ensure_cases)} ensure tests passed")

    if all_pass:
        print("\n\033[1;32m🎉 ALL CHECKS PASSED\033[0m")
    else:
        print("\n\033[1;31m❌ SOME CHECKS FAILED\033[0m")
