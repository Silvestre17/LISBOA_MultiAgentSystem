# ==========================================================================
# Master Thesis
#   - André Filipe Gomes Silvestre, 20240502
# 
# This module implements a web scraper for the "Visit Lisbon" PLACES page.
# It extracts details for various places (restaurants, museums, hotels, etc.)
# and saves the data in a structured JSON format.
# 
# Link to the places page: https://www.visitlisboa.com/en/places
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

def get_total_pages(session, headers, base_url):
    """
    Determines the total number of pages by inspecting the pagination control.
    
    Args:
        session (requests.Session): The requests session object.
        headers (dict): Headers to use for the HTTP request.
        base_url (str): The URL of the places listing page.
    
    Returns:
        int: The total number of pages. If unable to determine, returns 0.
    """
    print("Determining the total number of pages...")
    try:
        response = session.get(base_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        pagy_nav = soup.find('nav', id='pagy')
        if not pagy_nav:
            print("Pagination control not found. Assuming only 1 page.")
            return 1
            
        page_links = pagy_nav.find_all('a', href=True)
        page_numbers = [1]
        for link in page_links:
            if match := re.search(r'page=(\d+)', link['href']):
                page_numbers.append(int(match.group(1)))
        
        total_pages = max(page_numbers)
        print(f"Found a total of {total_pages} pages.")
        return total_pages
        
    except requests.exceptions.RequestException as e:
        print(f"Could not determine total pages due to an error: {e}. Aborting.")
        return 0
    except (ValueError, TypeError):
        print("Could not parse page numbers. Assuming 1 page.")
        return 1

def get_place_urls_from_page(session, page_number, headers, base_url):
    """
    Fetches a single page of place listings and extracts the URLs for each place.
    
    Args:
        session (requests.Session): The requests session object.
        page_number (int): The page number to fetch.
        headers (dict): Headers to use for the HTTP request.
        base_url (str): The base URL for the website.
    
    Returns:
        list or None: A list of place URLs if successful, None otherwise. If no places are found, returns an empty list.
    """
    list_page_url = f"{base_url}?page={page_number}"
    place_urls = []
    
    try:
        response = session.get(list_page_url, headers=headers)
        response.raise_for_status()
        time.sleep(random.uniform(1, 2))
        
        soup = BeautifulSoup(response.content, 'html.parser')
        cards = soup.find_all('div', attrs={'data-controller': 'clickable-card'})
        
        for card in cards:
            if link_tag := card.find('a', attrs={'data-clickable-card-target': 'link'}):
                if 'href' in link_tag.attrs:
                    
                    # Construct the full URL from the relative path
                    full_url = requests.compat.urljoin(base_url, link_tag['href']) # type: ignore
                    place_urls.append(full_url)
                
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching {list_page_url}: {e}")
        return None
        
    return place_urls

def scrape_place_details(session, place_url, headers):
    """
    Scrapes detailed information from a single place page.
    This function is designed to be flexible and handle various page layouts.
    
    Args:
        session (requests.Session): The requests session object.
        place_url (str): The URL of the place page to scrape.
        headers (dict): Headers to use for the HTTP request.
    
    Returns:
        dict or None: A dictionary containing place details if successful, None otherwise.
    """
    place_data = {'url': place_url}
    base_url = "https://www.visitlisboa.com"
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            response = session.get(place_url, headers=headers, timeout=10)
            if response.status_code == 429:
                print(f"  [Attempt {attempt + 1}/{max_retries}] Rate limited. Waiting {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # --- General Info ---
            if title_tag := soup.select_one('h1.font-serif, h2.max-w-xl'):
                 place_data['title'] = title_tag.get_text(strip=True)

            if category_tag := soup.find('div', class_='text-green-primary'):
                place_data['category'] = category_tag.get_text(strip=True)
            
            if desc_tag := soup.select_one('h2.max-w-xl + p, h1.font-serif + p'):
                place_data['short_description'] = desc_tag.get_text(strip=True)

            # --- Media ---
            place_data['image_urls'] = [requests.compat.urljoin(base_url, img['src']) for img in soup.select('div[data-carousel-target="track"] img') if 'src' in img.attrs] # type: ignore
            place_data['video_urls'] = [iframe['src'] for iframe in soup.find_all('iframe') if 'src' in iframe.attrs]

            # --- Full Description ---
            if details_div := soup.find('div', class_='from-cms'):
                place_data['full_description'] = details_div.get_text(separator='\n', strip=True)

            # --- Features (e.g., price range, Wi-Fi) ---
            place_data['features'] = [li.get_text(strip=True) for li in soup.select('ul.flex-wrap li.bg-green-primary')]
            
            # --- Contact, Location, Schedule ---
            place_data['contact_info'] = {}
            place_data['social_media'] = {}
            place_data['schedule'] = {}
            
            info_boxes = soup.find_all('div', class_='info-text')
            for box in info_boxes:
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
                            elif link.find('svg', {'class': 'select-none'}): # Heuristic for social media links
                                social_name = link['href']
                                try: # Try to get a better name from an icon
                                    use_tag = link.find('use')
                                    if use_tag and 'href' in use_tag.attrs:
                                        social_name = use_tag['href'].split('-')[0].replace('#icon', '')
                                except: pass
                                place_data['social_media'][social_name] = link.get('href')
                            else:
                                place_data['contact_info']['website'] = href
                    elif 'schedule' in h3_text:
                        if today := box.find('p'):
                            place_data['schedule']['today'] = today.get_text(strip=True)
                        hours_list = box.find_all('li')
                        for hour_item in hours_list:
                            day_span = hour_item.find('span', class_='flex-none')
                            if day_span:
                                time_span = day_span.find_next_sibling('span') # Find the sibling of the day span
                                if time_span: # Check if the time span exists before using it
                                    day_text = day_span.get_text(strip=True)
                                    time_text = time_span.get_text(strip=True)
                                    place_data['schedule'][day_text] = time_text
            
            # --- Tripadvisor Rating ---
            if reviews_section := soup.find('h2', string='Reviews'):
                if rating_div := reviews_section.find_next_sibling('div', class_='bg-off-white'):
                    place_data['tripadvisor'] = {}
                    if rating_val := rating_div.find('span', class_='font-bold'):
                        place_data['tripadvisor']['rating'] = rating_val.get_text(strip=True)
                    if reviews_count := rating_div.find('a', string=re.compile(r'reviews$')):
                        place_data['tripadvisor']['reviews_count'] = reviews_count.get_text(strip=True).replace(' reviews', '')
                        place_data['tripadvisor']['url'] = reviews_count['href']
            
            return place_data

        except requests.exceptions.RequestException as e:
            print(f"  [Attempt {attempt + 1}/{max_retries}] Error: {e}. Retrying...")
            time.sleep(retry_delay)
    
    print(f"  Failed to scrape {place_url} after {max_retries} attempts.")
    return None

def main():
    """
    Main function to orchestrate the scraping process for places.
    This function handles updating existing places, adding new ones,
    and removing places that are no longer listed on the website.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filepath = os.path.join(script_dir, 'places.json')
    base_url = "https://www.visitlisboa.com/en/places"

    # Load existing places from JSON file
    existing_places = {}
    if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
        print(f"Loading existing places from: {output_filepath}")
        with open(output_filepath, 'r', encoding='utf-8') as f:
            try:
                places_list = json.load(f)
                existing_places = {place['url']: place for place in places_list}
                print(f"Found {len(existing_places)} existing places.")
            except json.JSONDecodeError:
                print("Warning: Could not read existing JSON file. Starting fresh.")

    # --- Scrape all current place URLs from the website ---
    all_scraped_urls = set()
    with requests.Session() as session:
        total_pages = get_total_pages(session, headers, base_url)
        if total_pages == 0:
            return

        print("\nScraping all place URLs from the website...")
        for page in tqdm(range(1, total_pages + 1), desc="Scraping URLs", unit="page"):
            urls = get_place_urls_from_page(session, page, headers, base_url)
            if urls:
                all_scraped_urls.update(urls)
    
    print(f"Found {len(all_scraped_urls)} unique place URLs on the website.")

    # --- Process places: identify new, updated, and removed ---
    new_places = []
    updated_places = []
    unchanged_places = []
    
    existing_urls = set(existing_places.keys())
    
    # URLs for places that are currently on the website
    scraped_urls_set = all_scraped_urls
    
    # URLs for places that are new
    new_urls = scraped_urls_set - existing_urls
    
    # URLs for places that might be updated or are unchanged
    potentially_updated_urls = scraped_urls_set.intersection(existing_urls)
    
    # URLs for places that have been removed
    removed_urls = existing_urls - scraped_urls_set

    with requests.Session() as session:
        # Scrape new places
        if new_urls:
            print(f"\nScraping {len(new_urls)} new places...")
            for url in tqdm(new_urls, desc="Scraping new places", unit="place"):
                details = scrape_place_details(session, url, headers)
                if details:
                    new_places.append(details)
                time.sleep(random.uniform(1, 2))

        # Check for updates in existing places
        if potentially_updated_urls:
            print(f"\nChecking {len(potentially_updated_urls)} existing places for updates...")
            for url in tqdm(potentially_updated_urls, desc="Checking for updates", unit="place"):
                current_details = scrape_place_details(session, url, headers)
                if current_details:
                    # Normalize data for comparison by loading and dumping
                    existing_place_json = json.dumps(existing_places[url], sort_keys=True)
                    current_details_json = json.dumps(current_details, sort_keys=True)

                    if existing_place_json != current_details_json:
                        print(f"  - Place has been updated: {url}")
                        updated_places.append(current_details)
                    else:
                        unchanged_places.append(existing_places[url])
                else:
                    # If scraping fails, assume it's unchanged to avoid data loss
                    unchanged_places.append(existing_places[url])
                time.sleep(random.uniform(1, 2))
        else:
            # If no overlap, all existing places that are not removed are unchanged
            unchanged_urls = existing_urls - removed_urls
            for url in unchanged_urls:
                unchanged_places.append(existing_places[url])


    # --- Consolidate data and save ---
    final_place_list = unchanged_places + new_places + updated_places
    
    print("\n--- Synchronization Report ---")
    print(f"  - Added: {len(new_places)} new places.")
    print(f"  - Updated: {len(updated_places)} places.")
    print(f"  - Removed: {len(removed_urls)} places.")
    print(f"  - Unchanged: {len(unchanged_places)} places.")
    print(f"  - Total places to be saved: {len(final_place_list)}")

    # Save the updated list to the JSON file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(final_place_list, f, indent=4, ensure_ascii=False)
    
    print(f"\nSuccessfully saved {len(final_place_list)} places to {output_filepath}")

if __name__ == "__main__":
    main()