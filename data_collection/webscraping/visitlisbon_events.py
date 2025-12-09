# ==========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
# 
# This module implements a web scraper for the "Visit Lisbon" events page.
# It extracts event details such as title, description, date, price, and location,
# and saves the data in a structured JSON format.
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
from tqdm import tqdm               # To show progress bars

def get_total_pages(session, headers):
    """
    Determines the total number of event pages by inspecting the pagination control.
    
    Args:
        session (requests.Session): The requests session object.
        headers (dict): Headers to use for the HTTP request.
    
    Returns:
        int: The total number of pages. If unable to determine, returns 0.
    """
    base_url = "https://www.visitlisboa.com/en/events"
    print("Determining the total number of pages...")
    try:
        response = session.get(base_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the pagination navigation bar
        pagy_nav = soup.find('nav', id='pagy')
        if not pagy_nav:
            print("Pagination control not found. Assuming only 1 page.")
            return 1
            
        # Find all links within the pagination bar
        page_links = pagy_nav.find_all('a', href=True)
        
        page_numbers = [1] # Start with 1 in case there's only one page
        for link in page_links:
            # Use regex to find numbers in the href attribute
            if match := re.search(r'page=(\d+)', link['href']):
                page_numbers.append(int(match.group(1)))
        
        total_pages = max(page_numbers)
        print(f"Found a total of {total_pages} pages.")
        return total_pages
        
    except requests.exceptions.RequestException as e:
        print(f"Could not determine total pages due to an error: {e}. Aborting.")
        return 0
    except (ValueError, TypeError):
        print("Could not parse page numbers from pagination. Assuming 1 page.")
        return 1

def get_event_urls_from_page(session, page_number, headers):
    """
    Fetches a single page of event listings and extracts the URLs for each event.
    
    Args:
        session (requests.Session): The requests session object.
        page_number (int): The page number to fetch.
        headers (dict): Headers to use for the HTTP request.
    
    Returns:
        list or None: A list of event URLs if successful, None otherwise. If no events are found, returns an empty list.
    """
    base_url = "https://www.visitlisboa.com"
    list_page_url = f"{base_url}/en/events?page={page_number}"
    event_urls = []
    
    try:
        response = session.get(list_page_url, headers=headers)
        response.raise_for_status()
        time.sleep(random.uniform(1, 2)) # Polite delay
        
        soup = BeautifulSoup(response.content, 'html.parser')
        event_cards = soup.find_all('div', attrs={'data-controller': 'clickable-card'})
        
        if not event_cards:
            return []
            
        for card in event_cards:
            link_tag = card.find('a', attrs={'data-clickable-card-target': 'link'})
            if link_tag and 'href' in link_tag.attrs:
                event_urls.append(f"{base_url}{link_tag['href']}")
                
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching {list_page_url}: {e}")
        return None
        
    return event_urls

def scrape_event_details(session, event_url, headers):
    """
    Scrapes detailed information from a single event page with a retry mechanism.
    This version is heavily updated to extract more specific details.
    
    Args:
        session (requests.Session): The requests session object.
        event_url (str): The URL of the event page to scrape.
        headers (dict): Headers to use for the HTTP request.
    
    Returns:
        dict or None: A dictionary containing event details if successful, None otherwise.
    """
    event_data = {'url': event_url}
    base_url = "https://www.visitlisboa.com"
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            response = session.get(event_url, headers=headers)
            if response.status_code == 429:
                print(f"  [Attempt {attempt + 1}/{max_retries}] Rate limited. Waiting {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            
            # --- General Info ---
            if title_tag := soup.find('h1'):
                event_data['title'] = title_tag.get_text(strip=True)
            if category_tag := soup.find('div', class_='text-green-primary'):
                event_data['category'] = category_tag.get_text(strip=True)

            # --- Main Short Description ---
            if h2_title := soup.find('h2', class_='max-w-xl'):
                if short_desc_tag := h2_title.find_next_sibling('p'):
                    event_data['short_description'] = short_desc_tag.get_text(strip=True)

            # --- Image & Video URLs ---
            event_data['image_urls'] = []
            if carousel := soup.find('div', attrs={'data-carousel-target': 'track'}):
                images = carousel.find_all('img')
                for img in images:
                    if 'src' in img.attrs and img['src']:
                        event_data['image_urls'].append(f"{base_url}{img['src']}")

            event_data['video_urls'] = []
            iframes = soup.find_all('iframe')
            for iframe in iframes:
                if 'src' in iframe.attrs and iframe['src']:
                    event_data['video_urls'].append(iframe['src'])
            
            # --- Detailed Description ---
            if details_div := soup.find('div', class_='from-cms'):
                event_data['full_description'] = details_div.get_text(strip=True, separator='\n')

            # --- Dates & Times ---
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

            # --- Price / Entry Fee ---
            # Search for the specific span with a ticket icon first, then a generic one
            if price_span := soup.find('span', class_='bg-yellow-t60'):
                event_data['price'] = price_span.get_text(strip=True)
            elif price_span_generic := soup.find('span', string=lambda t: t and ('Free Entry' in t or 'From' in t)):
                event_data['price'] = price_span_generic.get_text(strip=True)


            # --- Location and Information ---
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
            
            return event_data # Success

        except requests.exceptions.RequestException as e:
            print(f"  [Attempt {attempt + 1}/{max_retries}] Error: {e}. Retrying...")
            time.sleep(retry_delay)
    
    print(f"  Failed to scrape {event_url} after {max_retries} attempts.")
    return None

def main():
    """
    Main function to orchestrate the scraping process.
    This function handles updating existing events, adding new ones,
    and removing events that are no longer listed on the website.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filepath = os.path.join(script_dir, 'events.json')
    
    # Load existing events from JSON file
    existing_events = {}
    if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
        print(f"Loading existing events from: {output_filepath}")
        with open(output_filepath, 'r', encoding='utf-8') as f:
            try:
                events_list = json.load(f)
                existing_events = {event['url']: event for event in events_list}
                print(f"Found {len(existing_events)} existing events.")
            except json.JSONDecodeError:
                print("Warning: Could not read existing JSON file. Starting fresh.")

    # --- Scrape all current event URLs from the website ---
    all_scraped_urls = set()
    with requests.Session() as session:
        total_pages = get_total_pages(session, headers)
        if total_pages == 0:
            return

        print("\nScraping all event URLs from the website...")
        for page in tqdm(range(1, total_pages + 1), desc="Scraping URLs", unit="page"):
            urls = get_event_urls_from_page(session, page, headers)
            if urls:
                all_scraped_urls.update(urls)
    
    print(f"Found {len(all_scraped_urls)} unique event URLs on the website.")

    # --- Process events: identify new, updated, and removed ---
    new_events = []
    updated_events = []
    unchanged_events = []
    
    existing_urls = set(existing_events.keys())
    
    # URLs for events that are currently on the website
    scraped_urls_set = all_scraped_urls
    
    # URLs for events that are new
    new_urls = scraped_urls_set - existing_urls
    
    # URLs for events that might be updated or are unchanged
    potentially_updated_urls = scraped_urls_set.intersection(existing_urls)
    
    # URLs for events that have been removed
    removed_urls = existing_urls - scraped_urls_set

    with requests.Session() as session:
        # Scrape new events
        if new_urls:
            print(f"\nScraping {len(new_urls)} new events...")
            for url in tqdm(new_urls, desc="Scraping new events", unit="event"):
                details = scrape_event_details(session, url, headers)
                if details:
                    new_events.append(details)
                time.sleep(random.uniform(1, 2))

        # Check for updates in existing events
        if potentially_updated_urls:
            print(f"\nChecking {len(potentially_updated_urls)} existing events for updates...")
            for url in tqdm(potentially_updated_urls, desc="Checking for updates", unit="event"):
                current_details = scrape_event_details(session, url, headers)
                if current_details:
                    # Normalize data for comparison by loading and dumping
                    # This avoids issues with float precision, key order, etc.
                    existing_event_json = json.dumps(existing_events[url], sort_keys=True)
                    current_details_json = json.dumps(current_details, sort_keys=True)

                    if existing_event_json != current_details_json:
                        print(f"  - Event has been updated: {url}")
                        updated_events.append(current_details)
                    else:
                        unchanged_events.append(existing_events[url])
                else:
                    # If scraping fails, assume it's unchanged to avoid data loss
                    unchanged_events.append(existing_events[url])
                time.sleep(random.uniform(1, 2))
        else:
            # If no overlap, all existing events that are not removed are unchanged
            unchanged_urls = existing_urls - removed_urls
            for url in unchanged_urls:
                unchanged_events.append(existing_events[url])


    # --- Consolidate data and save ---
    final_event_list = unchanged_events + new_events + updated_events
    
    print("\n--- Synchronization Report ---")
    print(f"  - Added: {len(new_events)} new events.")
    print(f"  - Updated: {len(updated_events)} events.")
    print(f"  - Removed: {len(removed_urls)} events.")
    print(f"  - Unchanged: {len(unchanged_events)} events.")
    print(f"  - Total events to be saved: {len(final_event_list)}")

    # Save the updated list to the JSON file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(final_event_list, f, indent=4, ensure_ascii=False)
    
    print(f"\nSuccessfully saved {len(final_event_list)} events to {output_filepath}")


if __name__ == "__main__":
    main()