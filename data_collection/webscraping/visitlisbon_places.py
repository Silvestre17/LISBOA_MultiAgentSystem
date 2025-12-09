# ==========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 2025
# 
# Description:
#   This module implements a robust web scraper for "Visit Lisbon" places
#   (museums, landmarks, etc.). It extracts details like schedule, contacts,
#   and location, managing incremental updates to a JSON file.
# 
#   Link: https://www.visitlisboa.com/en/places
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
    """Determines total pages for places.
    
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

def get_place_urls_from_page(session, page_number, base_url):
    """Fetches URLs from a specific page.
    
    Arg:
        session (requests.Session): The requests session.
        page_number (int): The page number to fetch.
        base_url (str): The base URL.
        
    Returns:
        list: List of place URLs. If error, returns empty list.
    """
    list_url = f"{base_url}?page={page_number}"
    place_urls = []
    
    try:
        time.sleep(random.uniform(2, 4)) # Stealth delay
        response = session.get(list_url, headers=get_headers(), timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        cards = soup.find_all('div', attrs={'data-controller': 'clickable-card'})
        
        for card in cards:
            if link_tag := card.find('a', attrs={'data-clickable-card-target': 'link'}):
                if 'href' in link_tag.attrs:
                    # Construct absolute URL
                    full_url = requests.compat.urljoin("https://www.visitlisboa.com", link_tag['href']) # type: ignore
                    place_urls.append(full_url)
                    
    except Exception as e:
        logging.error(f"Error fetching page {page_number}: {e}")
        
    return place_urls

def scrape_place_details(session, place_url):
    """
    Scrapes detailed info from a place page.
    
    Arg:
        session (requests.Session): The requests session.
        place_url (str): The URL of the place page to scrape.
        
    Returns:
        dict: Dictionary with place details. If error, returns None.
    """
    place_data = {'url': place_url}
    base_domain = "https://www.visitlisboa.com"
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(1.5, 3))
            
            response = session.get(place_url, headers=get_headers(), timeout=30)
            
            if response.status_code == 429:
                wait_time = (attempt + 2) * 5
                logging.warning(f"Rate limit on {place_url}. Waiting {wait_time}s")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # --- Parsing Logic ---
            
            # General
            if title := soup.select_one('h1.font-serif, h2.max-w-xl'):
                place_data['title'] = title.get_text(strip=True)
            if cat := soup.find('div', class_='text-green-primary'):
                place_data['category'] = cat.get_text(strip=True)
            if desc := soup.select_one('h2.max-w-xl + p, h1.font-serif + p'):
                place_data['short_description'] = desc.get_text(strip=True)
                
            # Media
            place_data['image_urls'] = [
                requests.compat.urljoin(base_domain, img['src'])  # type: ignore
                for img in soup.select('div[data-carousel-target="track"] img') 
                if 'src' in img.attrs
            ]
            place_data['video_urls'] = [
                iframe['src'] for iframe in soup.find_all('iframe') if 'src' in iframe.attrs
            ]
            
            # Full Description
            if details := soup.find('div', class_='from-cms'):
                place_data['full_description'] = details.get_text(separator='\n', strip=True)
                
            # Features
            place_data['features'] = [
                li.get_text(strip=True) for li in soup.select('ul.flex-wrap li.bg-green-primary')
            ]
            
            # Contacts, Location, Schedule
            place_data['contact_info'] = {}
            place_data['social_media'] = {}
            place_data['schedule'] = {}
            
            for box in soup.find_all('div', class_='info-text'):
                if h3 := box.find('h3'):
                    h3_text = h3.get_text(strip=True).lower()
                    
                    if 'location' in h3_text:
                        if content := box.find('div', class_='info-text__content'):
                            place_data['location'] = content.get_text(strip=True)
                            
                    elif 'information' in h3_text:
                        for link in box.find_all('a', href=True):
                            href = link['href']
                            if href.startswith('tel:'):
                                place_data['contact_info']['phone'] = href.replace('tel:', '').strip()
                            elif href.startswith('mailto:'):
                                place_data['contact_info']['email'] = href.replace('mailto:', '').strip()
                            elif link.find('svg', {'class': 'select-none'}):
                                social_name = href
                                # Attempt to extract readable name from icon
                                try:
                                    if use_tag := link.find('use'):
                                        social_name = use_tag['href'].split('-')[0].replace('#icon', '')
                                except: pass
                                place_data['social_media'][social_name] = href
                            else:
                                place_data['contact_info']['website'] = href
                                
                    elif 'schedule' in h3_text:
                        if today := box.find('p'):
                            place_data['schedule']['today'] = today.get_text(strip=True)
                        for li in box.find_all('li'):
                            day_span = li.find('span', class_='flex-none')
                            if day_span:
                                time_span = day_span.find_next_sibling('span')
                                if time_span:
                                    place_data['schedule'][day_span.get_text(strip=True)] = time_span.get_text(strip=True)

            # TripAdvisor Ratings (if available)
            if reviews_section := soup.find('h2', string='Reviews'):
                if rating_div := reviews_section.find_next_sibling('div', class_='bg-off-white'):
                    place_data['tripadvisor'] = {}
                    if val := rating_div.find('span', class_='font-bold'):
                        place_data['tripadvisor']['rating'] = val.get_text(strip=True)
                    if count_link := rating_div.find('a', string=re.compile(r'reviews$')):
                        place_data['tripadvisor']['reviews_count'] = count_link.get_text(strip=True).replace(' reviews', '')
                        place_data['tripadvisor']['url'] = count_link['href']

            return place_data

        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt+1} failed for {place_url}: {e}")
            time.sleep(2)

    logging.error(f"Failed to scrape {place_url}")
    return None

def main():
    """
    Main function to orchestrate the scraping process for places.
    This function handles updating existing places, adding new ones,
    and removing places that are no longer listed on the website.
    """    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filepath = os.path.join(script_dir, 'places.json')
    base_url = "https://www.visitlisboa.com/en/places"
    
    # 1. Load Existing
    existing_places = {}
    if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
        logging.info("Loading existing places...")
        try:
            with open(output_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                existing_places = {item['url']: item for item in data}
                logging.info(f"Loaded {len(existing_places)} existing places.")
        except json.JSONDecodeError:
            logging.warning("JSON corrupted. Starting fresh.")

    # 2. Scrape URLs
    all_urls = set()
    with requests.Session() as session:
        total_pages = get_total_pages(session, base_url)
        logging.info("Harvesting URLs...")
        
        for page in tqdm(range(1, total_pages + 1), desc="Pages", mininterval=5):
            urls = get_place_urls_from_page(session, page, base_url)
            all_urls.update(urls)

    logging.info(f"Total places found online: {len(all_urls)}")

    # --- SAFETY CHECK ---
    if len(all_urls) == 0:
        logging.error("CRITICAL: No places found! Possible blocking or site structure change. Aborting save.")
        sys.exit(1)
    
    if len(existing_places) > 0 and len(all_urls) < len(existing_places) * 0.5:
        logging.warning("WARNING: Significant drop in places count. Verify manually.")

    # 3. Delta Logic
    existing_urls = set(existing_places.keys())
    new_urls = all_urls - existing_urls
    removed_urls = existing_urls - all_urls
    potential_updates = all_urls.intersection(existing_urls)
    updated_urls = set()
    unchanged_urls = set()
    
    final_list = []

    with requests.Session() as session:
        # A. New
        if new_urls:
            logging.info(f"Scraping {len(new_urls)} new places.")
            for url in tqdm(new_urls, desc="New Places", mininterval=5):
                if details := scrape_place_details(session, url):
                    final_list.append(details)
        
        # B. Updated
        if potential_updates:
            logging.info(f"Checking {len(potential_updates)} existing places.")
            for url in tqdm(potential_updates, desc="Existing Places", mininterval=5):
                if new_details := scrape_place_details(session, url):
                    old_json = json.dumps(existing_places[url], sort_keys=True)
                    new_json = json.dumps(new_details, sort_keys=True)
                    
                    if old_json != new_json:
                        final_list.append(new_details)
                        updated_urls.add(url)
                        logging.info(f"Place {url} has been updated.")
                    else:
                        final_list.append(existing_places[url])
                        unchanged_urls.add(url)
                else:
                    final_list.append(existing_places[url])

    # 4. Save
    if len(final_list) > 0:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_list, f, indent=4, ensure_ascii=False)
        logging.info(f"Successfully saved {len(final_list)} places.")
    else:
        logging.error("Final list is empty. Something went wrong. Not saving.")
        sys.exit(1)
    
    # 5. Log Report
    logging.info("\n--- Synchronization Report ---")
    logging.info(f"  - Added: {len(new_urls)} new places.")
    logging.info(f"  - Updated: {len(updated_urls)} places.")
    logging.info(f"  - Removed: {len(removed_urls)} places.")
    logging.info(f"  - Unchanged: {len(unchanged_urls)} places.")
    logging.info(f"  - Total places to be saved: {len(final_list)}")
    logging.info("Places synchronization complete.")

if __name__ == "__main__":
    main()