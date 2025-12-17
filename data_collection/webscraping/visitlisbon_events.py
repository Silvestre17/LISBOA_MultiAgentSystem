# ==========================================================================
# Master Thesis - Web Scraper for "Visit Lisbon" Events
#   - André Filipe Gomes Silvestre, 20240502
# 
# Description:
#   This module implements a robust web scraper for "Visit Lisbon" EVENTS.
#   It extracts event details such as title, description, date, price, and location,
#   managing incremental updates to a JSON file.
# 
# Link to the events page: https://www.visitlisboa.com/en/events
# ==========================================================================

# Required libraries:
# pip install requests beautifulsoup4 tqdm

import requests                     # To make HTTP requests
from bs4 import BeautifulSoup       # To parse HTML content
import json                         # To handle JSON data
import time                         # To add delays  
import random                       # To make delays random
import os                           # To handle file paths correctly
import re                           # To extract numbers from strings
import logging                      # To log messages (for Github Actions)
import sys                          # To exit the script in case of critical errors
from tqdm import tqdm               # To show progress bars

# --- Configuration & Anti-Bot Measures ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
]

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
        time.sleep(random.uniform(2, 4)) # Stealth delay
        response = session.get(list_url, headers=get_headers(), timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        event_cards = soup.find_all('div', attrs={'data-controller': 'clickable-card'})
        
        for card in event_cards:
            if link_tag := card.find('a', attrs={'data-clickable-card-target': 'link'}):
                if 'href' in link_tag.attrs:
                    # Construct absolute URL
                    full_url = requests.compat.urljoin("https://www.visitlisboa.com", link_tag['href']) # type: ignore
                    event_urls.append(full_url)
                    
    except Exception as e:
        logging.error(f"Error fetching page {page_number}: {e}")
        
    return event_urls

def scrape_event_details(session, event_url):
    """
    Scrapes detailed info from an event page.
    
    Arg:
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
            if title_tag := soup.find('h1'):
                event_data['title'] = title_tag.get_text(strip=True)
            if category_tag := soup.find('div', class_='text-green-primary'):
                event_data['category'] = category_tag.get_text(strip=True)

            # Main Short Description
            if h2_title := soup.find('h2', class_='max-w-xl'):
                if short_desc_tag := h2_title.find_next_sibling('p'):
                    event_data['short_description'] = short_desc_tag.get_text(strip=True)

            # Image & Video URLs
            event_data['image_urls'] = []
            if carousel := soup.find('div', attrs={'data-carousel-target': 'track'}):
                images = carousel.find_all('img')
                for img in images:
                    if 'src' in img.attrs and img['src']:
                        event_data['image_urls'].append(requests.compat.urljoin(base_domain, img['src'])) # type: ignore

            event_data['video_urls'] = []
            iframes = soup.find_all('iframe')
            for iframe in iframes:
                if 'src' in iframe.attrs and iframe['src']:
                    event_data['video_urls'].append(iframe['src'])
            
            # Detailed Description
            if details_div := soup.find('div', class_='from-cms'):
                event_data['full_description'] = details_div.get_text(strip=True, separator='\n')

            # Dates & Times
            event_data['dates'] = []
            main_date_container = soup.find('div', class_='flex-wrap gap-4 mt-2')
            if main_date_container:
                times = main_date_container.find_all('time')
                if len(times) == 2:
                    event_data['dates'].append({'start': times[0].get_text(strip=True), 'end': times[1].get_text(strip=True)})
                elif len(times) == 1:
                    event_data['dates'].append({'start': times[0].get_text(strip=True), 'end': None})

            if more_dates_section := soup.find('div', id='dates'):
                date_divs = more_dates_section.find_all('div', class_='border-b')
                for div in date_divs:
                    times = div.find_all('time')
                    if len(times) == 2:
                        event_data['dates'].append({'start': times[0].get_text(strip=True), 'end': times[1].get_text(strip=True)})
                    elif len(times) == 1:
                        event_data['dates'].append({'start': times[0].get_text(strip=True), 'end': None})

            # Price / Entry Fee
            if price_span := soup.find('span', class_='bg-yellow-t60'):
                event_data['price'] = price_span.get_text(strip=True)
            elif price_span_generic := soup.find('span', string=lambda t: t and ('Free Entry' in t or 'From' in t)):
                event_data['price'] = price_span_generic.get_text(strip=True)

            # Location and Information
            info_boxes = soup.find_all('div', class_='info-text')
            event_data['information_links'] = {}
            for box in info_boxes:
                if h3 := box.find('h3'):
                    h3_text = h3.get_text().strip()
                    # Check for location keywords
                    if 'Address' in h3_text or 'Avenida' in h3_text or 'Parque' in h3_text or h3_text == "Estádio da Luz":
                         if content := box.find('div', class_='info-text__content'):
                            event_data['location'] = content.get_text(strip=True)
                    elif 'Information' in h3_text:
                        links = box.find_all('a')
                        for link in links:
                            link_text = link.get_text(strip=True)
                            if 'href' in link.attrs:
                                event_data['information_links'][link_text] = link['href']

            return event_data

        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt+1} failed for {event_url}: {e}")
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
        logging.warning("WARNING: Significant drop in events count. Verify manually.")

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
    logging.info(f"  - Unchanged: {len(unchanged_urls)} events.")
    logging.info(f"  - Total events to be saved: {len(final_list)}")
    logging.info("Events synchronization complete.")

if __name__ == "__main__":
    main()