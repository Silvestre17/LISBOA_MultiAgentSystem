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
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filepath = os.path.join(script_dir, 'places.json')
    base_url = "https://www.visitlisboa.com/en/places"

    all_places_data = []
    if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
        print(f"Resuming from existing file: {output_filepath}")
        with open(output_filepath, 'r', encoding='utf-8') as f:
            try:
                all_places_data = json.load(f)
            except json.JSONDecodeError:
                all_places_data = []

    scraped_place_urls = {place.get('url') for place in all_places_data}
    print(f"Found {len(scraped_place_urls)} places already scraped.")

    with requests.Session() as session:
        total_pages = get_total_pages(session, headers, base_url)
        if total_pages == 0:
            return
        
        # Iterate through each page and scrape place URLs with tqdm progress bar
        for page in tqdm(range(1, total_pages + 1), desc="Total Page Progress", unit="page"):
            print(f"\n--- Scraping Page {page}/{total_pages} ---")
            place_urls = get_place_urls_from_page(session, page, headers, base_url)
            
            if place_urls is None:
                print(f"Could not fetch URLs from page {page}. Skipping.")
                continue               

            new_places_on_page = []
            for url in place_urls:
                if url not in scraped_place_urls:
                    print(f"  - Scraping new place: {url}")
                    details = scrape_place_details(session, url, headers)
                    if details:
                        new_places_on_page.append(details)
                        scraped_place_urls.add(url)
                    time.sleep(random.uniform(1, 2.5)) # Polite delay
                else:
                    print(f"  - Skipping already scraped place: {url}")

            if new_places_on_page:
                all_places_data.extend(new_places_on_page)
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(all_places_data, f, indent=4, ensure_ascii=False)
                print(f"  >> Saved {len(new_places_on_page)} new places. Total saved: {len(all_places_data)}.")
            else:
                print("  No new places found on this page (all previously scraped).")

    print(f"\nScraping complete. Total unique places in file: {len(all_places_data)}.")

if __name__ == "__main__":
    main()