# ==========================================================================
# Master Thesis - Web Scraper for "Visit Lisbon" Events
#   - André Filipe Gomes Silvestre, 20240502
#
# Description:
#   This module implements a robust web scraper for "Visit Lisbon" EVENTS.
#   It extracts event details such as title, description, date, price, and location,
#   managing incremental updates to a JSON file.
#
# Usage:
#   > python data_collection/webscraping/visitlisbon_events.py
#       Scrape the VisitLisboa events catalogue and update `data_collection/webscraping/events.json`.
#
# Notes:
#   - The script merges by URL, adds new events, updates changed ones, and removes entries no longer listed online.
#   - Saving is aborted if the scraper finds zero events, to avoid wiping the dataset after a blocking or markup failure.
#
# Link to the events page: https://www.visitlisboa.com/en/events
# ==========================================================================

# Required libraries:
# pip install requests beautifulsoup4 tqdm

import json                     # To handle JSON data
import logging                  # To log messages (for Github Actions)
import os                       # To handle file paths correctly
import random                   # To make delays random
import re                       # To extract numbers from strings
import sys                      # To exit the script in case of critical errors
import time                     # To add delays
from datetime import datetime   # To handle date parsing and formatting

import requests                 # To make HTTP requests
from bs4 import BeautifulSoup   # To parse HTML content
from tqdm import tqdm           # To show progress bars

# --- Configuration & Anti-Bot Measures ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
]

MONTH_NAME_TO_NUMBER = {
    'jan': 1,
    'january': 1,
    'feb': 2,
    'february': 2,
    'mar': 3,
    'march': 3,
    'apr': 4,
    'april': 4,
    'may': 5,
    'jun': 6,
    'june': 6,
    'jul': 7,
    'july': 7,
    'aug': 8,
    'august': 8,
    'sep': 9,
    'sept': 9,
    'september': 9,
    'oct': 10,
    'october': 10,
    'nov': 11,
    'november': 11,
    'dec': 12,
    'december': 12,
}


def _normalize_text(text):
    """Normalizes whitespace while preserving readable text for scraping."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', str(text).replace('\xa0', ' ')).strip()


def _first_element(soup, selectors):
    """Returns the first matching element for a list of selectors."""
    for selector in selectors:
        if element := soup.select_one(selector):
            return element
    return None


def _infer_iso_date(day_text, month_text, year_text=None, fallback_year=None):
    """Converts textual month/day/year pieces into ISO date format."""
    month_number = MONTH_NAME_TO_NUMBER.get((month_text or '').strip().lower())
    if not month_number:
        return None

    day = int(day_text)
    year = int(year_text) if year_text else int(fallback_year or datetime.now().year)
    try:
        return datetime(year, month_number, day).strftime('%Y-%m-%d')
    except ValueError:
        return None


def _append_unique_date_entry(date_entries, candidate):
    """Appends a date entry if it is not already present."""
    candidate_key = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
    if candidate_key not in {json.dumps(entry, sort_keys=True, ensure_ascii=False) for entry in date_entries}:
        date_entries.append(candidate)


def _extract_event_dates_from_text(text):
    """Parses date ranges or dated sessions from free text as a fallback."""
    normalized = _normalize_text(text)
    if not normalized:
        return []

    month_pattern = r'Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?'
    range_pattern = re.compile(
        rf'(?P<start_day>\d{{1,2}})\s+(?P<start_month>{month_pattern})(?:,?\s*(?P<start_year>\d{{4}}))?\s*[–-]\s*'
        rf'(?P<end_day>\d{{1,2}})\s+(?P<end_month>{month_pattern})(?:,?\s*(?P<end_year>\d{{4}}))?',
        re.IGNORECASE,
    )
    single_pattern = re.compile(
        rf'(?P<day>\d{{1,2}})\s+(?P<month>{month_pattern})(?:,?\s*(?P<year>\d{{4}}))?(?:\s*(?:\||at)?\s*(?P<time>\d{{1,2}}:\d{{2}}))',
        re.IGNORECASE,
    )

    extracted_dates = []

    for match in range_pattern.finditer(normalized):
        end_year = match.group('end_year')
        start_year = match.group('start_year') or end_year
        start_iso = _infer_iso_date(
            match.group('start_day'),
            match.group('start_month'),
            start_year,
            fallback_year=end_year,
        )
        end_iso = _infer_iso_date(
            match.group('end_day'),
            match.group('end_month'),
            end_year,
            fallback_year=start_year,
        )
        if start_iso and end_iso:
            _append_unique_date_entry(
                extracted_dates,
                {
                    'type': 'range',
                    'start': {
                        'datetime_iso': start_iso,
                        'display_text': _normalize_text(match.group(0).split('–')[0].split('-')[0]),
                        'time': None,
                    },
                    'end': {
                        'datetime_iso': end_iso,
                        'display_text': _normalize_text(match.group(0).split('–')[-1].split('-')[-1]),
                        'time': None,
                    },
                },
            )

    for match in single_pattern.finditer(normalized):
        iso_date = _infer_iso_date(match.group('day'), match.group('month'), match.group('year'))
        if not iso_date:
            continue
        _append_unique_date_entry(
            extracted_dates,
            {
                'type': 'single',
                'date': {
                    'datetime_iso': iso_date,
                    'display_text': _normalize_text(
                        f"{match.group('day')} {match.group('month')}"
                        + (f", {match.group('year')}" if match.group('year') else "")
                    ),
                    'time': match.group('time'),
                },
            },
        )

    return extracted_dates


def _extract_event_schedule_notes(text):
    """Extracts recurring-session notes from free text when they are user-visible."""
    normalized_text = str(text or "").replace('\r', '\n')
    if not normalized_text.strip():
        return []

    weekday_pattern = re.compile(
        r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|every|daily|'
        r'segunda(?:\-feira)?|ter[c\u00e7]a(?:\-feira)?|quarta(?:\-feira)?|'
        r'quinta(?:\-feira)?|sexta(?:\-feira)?|s[a\u00e1]bado|domingo)\b',
        re.IGNORECASE,
    )
    time_pattern = re.compile(
        r'\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b'     # 9 a.m. / 9 p.m.
        r'|\b\d{1,2}:\d{2}\b'                                 # 21:00
        r'|\b\d{1,2}h(?:\d{2})?\b',                           # 21h / 21h30 (PT/FR)
        re.IGNORECASE,
    )

    notes = []
    seen = set()
    for raw_line in re.split(r'\n+', normalized_text):
        note = _normalize_text(raw_line)
        if not note:
            continue
        if weekday_pattern.search(note) and time_pattern.search(note):
            if note not in seen:
                seen.add(note)
                notes.append(note)

    return notes


def _extract_event_highlight_links(details_div, base_domain):
    """Extracts structured highlight links from aggregated event pages."""
    if not details_div:
        return []

    highlight_links = []
    seen_urls = set()
    for heading in details_div.find_all(re.compile(r'^h[1-6]$')):
        if _normalize_text(heading.get_text(" ", strip=True)).lower() != 'highlights':
            continue

        for sibling in heading.find_next_siblings():
            if sibling.name and re.fullmatch(r'h[1-6]', sibling.name, re.IGNORECASE):
                break
            for link in sibling.find_all('a', href=True):
                text = _normalize_text(link.get_text(" ", strip=True))
                href = requests.compat.urljoin(base_domain, link['href'])  # type: ignore
                if not text or href in seen_urls:
                    continue
                seen_urls.add(href)
                highlight_links.append({'title': text, 'url': href})

    return highlight_links


def get_headers():
    """Generates headers with random User-Agent.

    Arg:
        None

    Returns:
        dict: Headers dictionary.
    """
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/'
    }


def get_total_pages(session, base_url):
    """Determines total pages for events.

    Arg:
        session (requests.Session): The requests session.
        base_url (str): The base URL.

    Returns:
        int: Total number of pages. If error, returns 1.
    """
    logging.info("Determining total pages...")
    try:
        response = session.get(base_url, headers=get_headers(), timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        pagy_nav = soup.find('nav', id='pagy')
        if not pagy_nav:
            return 1

        page_numbers = [1]
        for link in pagy_nav.find_all('a', href=True):
            if match := re.search(r'page=(\d+)', link['href']):
                page_numbers.append(int(match.group(1)))

        logging.info(f"Found a total of {max(page_numbers)} pages.")
        return max(page_numbers)
    except Exception as e:
        logging.error(f"Error determining pages: {e}")
        return 1


def get_event_urls_from_page(session, page_number, base_url):
    """Fetches URLs from a specific page.

    Arg:
        session (requests.Session): The requests session.
        page_number (int): The page number to fetch.
        base_url (str): The base URL.

    Returns:
        list: List of event URLs. If error, returns empty list.
    """
    list_url = f"{base_url}?page={page_number}"
    event_urls = []

    try:
        time.sleep(random.uniform(2, 4))  # Stealth delay
        response = session.get(list_url, headers=get_headers(), timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        event_cards = soup.find_all('div', attrs={'data-controller': 'clickable-card'})

        for card in event_cards:
            if link_tag := card.find('a', attrs={'data-clickable-card-target': 'link'}):
                if 'href' in link_tag.attrs:
                    # Construct absolute URL
                    full_url = requests.compat.urljoin("https://www.visitlisboa.com", link_tag['href'])  # type: ignore
                    event_urls.append(full_url)

    except Exception as e:
        logging.error(f"Error fetching page {page_number}: {e}")

    return event_urls


def _extract_time_from_container(container):
    """
    Extracts time string from a container that has a schedule icon.

    Args:
        container: BeautifulSoup element containing time info.

    Returns:
        str or None: Time string if found.
    """
    # Look for time in span after schedule icon
    time_divs = container.find_all('div', class_='fill-current')
    for div in time_divs:
        # Check if this div contains a schedule icon (not calendar)
        svg_use = div.find('use')
        if svg_use and 'href' in svg_use.attrs:
            if 'schedule' in svg_use['href']:
                if time_span := div.find('span'):
                    return time_span.get_text(strip=True)
    return None


def _parse_date_entry(time_element, time_str=None):
    """
    Parses a single date entry from a time element.

    Args:
        time_element: BeautifulSoup <time> element.
        time_str (str, optional): Associated time string (e.g., "18:30").

    Returns:
        dict: Date entry with datetime_iso, display_text, and time fields.
    """
    return {
        'datetime_iso': time_element.get('datetime'),  # ISO format: "2026-01-04"
        'display_text': time_element.get_text(strip=True),  # "04 Jan, 2026"
        'time': time_str  # "18:30" or None
    }


def scrape_event_details(session, event_url):
    """
    Scrapes detailed info from an event page.

    Handles different page structures for dates:
    - Single date events (e.g., concerts)
    - Date range events (e.g., exhibitions)
    - Multi-date events with specific times (e.g., theater performances)

    Args:
        session (requests.Session): The requests session.
        event_url (str): The URL of the event page to scrape.

    Returns:
        dict: Dictionary with event details. If error, returns None.
    """
    event_data = {'url': event_url}
    base_domain = "https://www.visitlisboa.com"
    max_retries = 3

    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(1.5, 3))

            response = session.get(event_url, headers=get_headers(), timeout=30)

            if response.status_code == 429:
                wait_time = (attempt + 2) * 5
                logging.warning(f"Rate limit on {event_url}. Waiting {wait_time}s")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # --- Parsing Logic ---

            # General Info
            title_tag = _first_element(soup, ['h2.max-w-xl', 'h1.font-serif', 'main h2', 'main h1'])
            if title_tag:
                event_data['title'] = _normalize_text(title_tag.get_text(" ", strip=True))
            if category_tag := soup.find('div', class_='text-green-primary'):
                event_data['category'] = _normalize_text(category_tag.get_text(" ", strip=True))

            header_block = title_tag.find_parent('div') if title_tag else None

            # Main Short Description (paragraph after title h2)
            short_desc_tag = None
            if header_block:
                short_desc_tag = header_block.find_next_sibling('p')
            if not short_desc_tag:
                short_desc_tag = _first_element(soup, ['h2.max-w-xl + p', 'h1.font-serif + p', 'main p'])
            if short_desc_tag:
                short_description = _normalize_text(short_desc_tag.get_text(" ", strip=True))
                if short_description:
                    event_data['short_description'] = short_description

            # Image & Video URLs
            event_data['image_urls'] = []
            if carousel := soup.find('div', attrs={'data-carousel-target': 'track'}):
                images = carousel.find_all('img')
                for img in images:
                    if 'src' in img.attrs and img['src']:
                        event_data['image_urls'].append(requests.compat.urljoin(base_domain, img['src']))  # type: ignore

            event_data['video_urls'] = []
            iframes = soup.find_all('iframe')
            for iframe in iframes:
                if 'src' in iframe.attrs and iframe['src']:
                    event_data['video_urls'].append(iframe['src'])

            # Detailed Description
            if details_div := soup.find('div', class_='from-cms'):
                event_data['full_description'] = details_div.get_text(strip=True, separator='\n')
                highlight_links = _extract_event_highlight_links(details_div, base_domain)
                if highlight_links:
                    event_data['highlight_links'] = highlight_links
                schedule_notes = _extract_event_schedule_notes(event_data.get('full_description', ''))
                if schedule_notes:
                    event_data['schedule_notes'] = schedule_notes

            # =============================================
            # DATES & TIMES - Improved Parsing
            # =============================================
            event_data['dates'] = []

            # 1. Main date container (header section with date badge)
            main_date_container = None
            if header_block:
                main_date_container = header_block.find(lambda tag: tag.name == 'div' and tag.find('time'))
            if not main_date_container:
                main_date_container = _first_element(
                    soup,
                    [
                        'div time',
                    ],
                )
                main_date_container = main_date_container.parent if main_date_container else None

            if main_date_container:
                times_elements = main_date_container.find_all('time')
                header_time_str = _extract_time_from_container(main_date_container)

                if len(times_elements) == 2:
                    # Date RANGE (e.g., exhibitions: "12 Dec, 2024 - 31 Dec, 2025")
                    event_data['dates'].append({
                        'type': 'range',
                        'start': _parse_date_entry(times_elements[0]),
                        'end': _parse_date_entry(times_elements[1])
                    })
                elif len(times_elements) == 1:
                    # Single date (may or may not have time)
                    event_data['dates'].append({
                        'type': 'single',
                        'date': _parse_date_entry(times_elements[0], header_time_str)
                    })

            # 1B. Fallback from textual header/details if the structured date badge changed
            if not event_data['dates']:
                fallback_text_parts = []
                if header_block:
                    fallback_text_parts.append(header_block.get_text(" ", strip=True))
                if event_data.get('short_description'):
                    fallback_text_parts.append(event_data['short_description'])
                if event_data.get('full_description'):
                    fallback_text_parts.append(event_data['full_description'])
                event_data['dates'] = _extract_event_dates_from_text("\n".join(fallback_text_parts))

            # 2. Additional dates section (id="dates") - for multi-date events
            if more_dates_section := soup.find('div', id='dates'):
                # Find all date rows (border-b class)
                date_rows = more_dates_section.find_all('div', class_='border-b')

                for row in date_rows:
                    time_element = row.find('time')
                    if time_element:
                        # Extract time from this specific row
                        row_time_str = _extract_time_from_container(row)

                        date_entry = {
                            'type': 'single',
                            'date': _parse_date_entry(time_element, row_time_str)
                        }

                        # Avoid duplicates (header date might be repeated)
                        is_duplicate = False
                        for existing in event_data['dates']:
                            if existing.get('type') == 'single':
                                if (existing.get('date', {}).get('datetime_iso') == date_entry['date']['datetime_iso'] and
                                        existing.get('date', {}).get('time') == date_entry['date']['time']):
                                    is_duplicate = True
                                    break

                        if not is_duplicate:
                            event_data['dates'].append(date_entry)

            # =============================================
            # PRICE / Entry Fee
            # =============================================
            # Look for price in yellow badge
            price_container = _first_element(
                soup,
                [
                    'span[class*="bg-yellow"]',
                    'div[class*="bg-yellow"] span',
                    'div[class*="bg-yellow"]',
                ],
            )
            if price_container:
                event_data['price'] = _normalize_text(price_container.get_text(" ", strip=True))
            elif price_span_generic := soup.find('span', string=lambda t: t and ('Free Entry' in t or 'From' in t)):
                event_data['price'] = _normalize_text(price_span_generic.get_text(" ", strip=True))

            # =============================================
            # LOCATION and VENUE - Improved Parsing
            # =============================================
            info_boxes = soup.find_all('div', class_='info-text')
            event_data['information_links'] = {}
            event_data['buy_tickets_url'] = None
            event_data['venue_locations'] = []

            for box in info_boxes:
                if h3 := box.find('h3'):
                    h3_text = _normalize_text(h3.get_text(" ", strip=True))
                    h3_lower = h3_text.lower()
                    content_div = box.find('div', class_='info-text__content')

                    # Check if this is the Information box
                    if h3_lower == 'information':
                        if content_div:
                            links = content_div.find_all('a')
                            for link in links:
                                if 'href' in link.attrs:
                                    # Check for ticket links
                                    link_text = _normalize_text(link.get_text(" ", strip=True)) or _normalize_text(link['href'])
                                    if 'buy tickets' in link_text.lower() or 'ticket' in link['href'].lower():
                                        event_data['buy_tickets_url'] = link['href']
                                    else:
                                        event_data['information_links'][link_text] = link['href']

                    # Check if this is the Dates box (skip, handled above)
                    elif h3_lower == 'dates':
                        continue

                    # Otherwise, assume it's a VENUE/LOCATION box
                    else:
                        venue_payload = {'venue_name': _normalize_text(h3_text)}
                        if content_div:
                            venue_payload['location'] = _normalize_text(content_div.get_text(" ", strip=True))

                        if venue_payload not in event_data['venue_locations']:
                            event_data['venue_locations'].append(venue_payload)

                        if not event_data.get('venue_name'):
                            event_data['venue_name'] = venue_payload['venue_name']
                        if venue_payload.get('location') and not event_data.get('location'):
                            event_data['location'] = venue_payload['location']

            if not event_data.get('title') and event_data.get('url'):
                slug = event_data['url'].rstrip('/').split('/')[-1].replace('-', ' ')
                event_data['title'] = _normalize_text(slug.title())

            if event_data.get('price'):
                event_data['price'] = _normalize_text(event_data['price'])

            return event_data

        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed for {event_url}: {e}")
            time.sleep(2)

    logging.error(f"Failed to scrape {event_url}")
    return None


def main():
    """Main function to orchestrate the scraping process for events.
    This function handles updating existing events, adding new ones,
    and removing events that are no longer listed on the website.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filepath = os.path.join(script_dir, 'events.json')
    base_url = "https://www.visitlisboa.com/en/events"

    # 1. Load Existing
    existing_events = {}
    if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
        logging.info("Loading existing events...")
        try:
            with open(output_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                existing_events = {item['url']: item for item in data}
                logging.info(f"Loaded {len(existing_events)} existing events.")
        except json.JSONDecodeError:
            logging.warning("JSON corrupted. Starting fresh.")

    # 2. Scrape URLs
    all_urls = set()
    with requests.Session() as session:
        total_pages = get_total_pages(session, base_url)
        logging.info("Harvesting URLs...")

        for page in tqdm(range(1, total_pages + 1), desc="Pages", mininterval=5):
            urls = get_event_urls_from_page(session, page, base_url)
            all_urls.update(urls)

    logging.info(f"Total events found online: {len(all_urls)}")

    # --- SAFETY CHECK ---
    if len(all_urls) == 0:
        logging.error("CRITICAL: No events found! Possible blocking or site structure change. Aborting save.")
        sys.exit(1)

    if len(existing_events) > 0 and len(all_urls) < len(existing_events) * 0.5:
        logging.error(
            "CRITICAL: Harvested events dropped from %s to %s. "
            "Possible blocking or VisitLisboa schema drift. Aborting save to protect existing JSON.",
            len(existing_events),
            len(all_urls),
        )
        sys.exit(1)

    # 3. Delta Logic
    existing_urls = set(existing_events.keys())
    new_urls = all_urls - existing_urls
    removed_urls = existing_urls - all_urls
    potential_updates = all_urls.intersection(existing_urls)
    updated_urls = set()
    unchanged_urls = set()

    final_list = []

    with requests.Session() as session:
        # A. New
        if new_urls:
            logging.info(f"Scraping {len(new_urls)} new events.")
            for url in tqdm(new_urls, desc="New Events", mininterval=5):
                if details := scrape_event_details(session, url):
                    final_list.append(details)
                    logging.info(f"New event added: {url}")

        # B. Updated
        if potential_updates:
            logging.info(f"Checking {len(potential_updates)} existing events.")
            for url in tqdm(potential_updates, desc="Existing Events", mininterval=5):
                if new_details := scrape_event_details(session, url):
                    old_json = json.dumps(existing_events[url], sort_keys=True)
                    new_json = json.dumps(new_details, sort_keys=True)

                    if old_json != new_json:
                        final_list.append(new_details)
                        updated_urls.add(url)
                        logging.info(f"Event {url} has been updated.")
                    else:
                        final_list.append(existing_events[url])
                        unchanged_urls.add(url)
                else:
                    final_list.append(existing_events[url])

    # 4. Save
    if len(final_list) > 0:
        if len(existing_events) > 0 and len(final_list) < len(existing_events) * 0.5:
            logging.error(
                "CRITICAL: Final events list dropped from %s to %s after detail scraping. "
                "Aborting save to protect existing JSON.",
                len(existing_events),
                len(final_list),
            )
            sys.exit(1)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_list, f, indent=4, ensure_ascii=False)
        logging.info(f"Successfully saved {len(final_list)} events.")
    else:
        logging.error("Final list is empty. Something went wrong. Not saving.")
        sys.exit(1)

    # 5. Log Report
    logging.info("\n--- Synchronization Report ---")
    logging.info(f"  - Added: {len(new_urls)} new events.")
    logging.info(f"  - Updated: {len(updated_urls)} events.")
    logging.info(f"  - Removed: {len(removed_urls)} events.")
    if removed_urls:
        for url in removed_urls:
            logging.info(f"    Removed event: {url}")
    logging.info(f"  - Unchanged: {len(unchanged_urls)} events.")
    logging.info(f"  - Total events to be saved: {len(final_list)}")
    logging.info("Events synchronization complete.")


if __name__ == "__main__":
    main()
