# ==========================================================================
# Master Thesis - Web Scraper for dados.gov.pt (Lisbon Datasets)
#   - André Filipe Gomes Silvestre, 20240502
#
# This script scrapes the dados.gov.pt portal specifically for datasets
# related to the Municipality of Lisbon (1106).
# It extracts the title, description, and the "Stable URL" for each dataset.
# It filters out datasets with "Resultados" or "Desafio" in the title.
#
# Link to the portal: https://dados.gov.pt/pt/datasets/?geozone=pt%3Aconcelho%3A1106
# ==========================================================================

# Required libraries:
# pip install requests beautifulsoup4

import json                                     # For saving data in JSON format
import logging                                  # For logging information
import os                                       # For file path operations
import time                                     # For adding delays
from typing import Any, Dict, List, Optional    # For type hinting
from urllib.parse import urljoin                # For constructing absolute URLs

import requests                                 # For making HTTP requests
from bs4 import BeautifulSoup                   # For parsing HTML content
from tqdm import tqdm                           # For progress bars

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',

    # If we want to log to a file as well as console
    # handlers=[
    #     logging.FileHandler("scraper.log"),
    #     logging.StreamHandler()
    # ]
)


class LisbonOpenDataScraper:
    """
    A class to scrape dataset information from the Portuguese Open Data Portal (dados.gov.pt),
    specifically targeting the Lisbon municipality.

    Attributes:
        base_url (str): The base URL of the portal.
        search_url (str): The search endpoint with filters for Lisbon.
        session (requests.Session): The HTTP session for making requests.
    """

    def __init__(self):
        """
        Initializes the scraper with the base URL and sets up the HTTP session.
        """
        self.base_url = "https://dados.gov.pt"
        # Filter: geozone = pt:concelho:1106 (Lisboa)
        self.search_url = f"{self.base_url}/pt/datasets/?geozone=pt%3Aconcelho%3A1106"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,pt;q=0.8"
        })
        self.skipped_count = 0

    def _get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetches a URL and returns a BeautifulSoup object.
        Implements retries with exponential backoff for transient errors.

        Args:
            url (str): The URL to fetch.

        Returns:
            Optional[BeautifulSoup]: The parsed HTML content, or None if the request failed.
        """
        retries = 3
        backoff_factor = 2

        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return BeautifulSoup(response.content, 'html.parser')  # type: ignore

            except requests.exceptions.RequestException as e:
                wait_time = backoff_factor ** attempt
                logging.warning(f"Error fetching {url}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        logging.error(f"Failed to fetch {url} after {retries} attempts.")
        return None

    def extract_dataset_details(self, dataset_url: str) -> Dict[str, str]:
        """
        Visits a specific dataset page to extract detailed information,
        specifically the 'URL Estável', available file formats, and last update date.

        Args:
            dataset_url (str): The URL of the specific dataset page.

        Returns:
            Dict[str, str]: A dictionary containing the stable URL, full description, file formats, and last update.
        """
        soup = self._get_soup(dataset_url)
        if not soup:
            return {"stable_url": "N/A", "full_description": "N/A", "file_formats": "N/A", "last_updated": "N/A"}

        details = {
            "stable_url": "N/A",
            "full_description": "N/A",
            "file_formats": "N/A",
            "last_updated": "N/A"
        }

        # --- Strategy 1: JSON-LD (Structured Data) ---
        json_ld_script = soup.find("script", {"type": "application/ld+json", "id": "json_ld"})
        if json_ld_script:
            try:
                data = json.loads(json_ld_script.string)

                # Extract Last Updated
                if "dateModified" in data:
                    details["last_updated"] = data["dateModified"]
                elif "dateCreated" in data:
                    details["last_updated"] = data["dateCreated"]

                # Extract File Formats
                if "distribution" in data:
                    formats = set()
                    for dist in data["distribution"]:
                        if "encodingFormat" in dist:
                            formats.add(dist["encodingFormat"])
                        elif "fileFormat" in dist:
                            formats.add(dist["fileFormat"])

                    if formats:
                        details["file_formats"] = ", ".join(sorted(formats))

            except json.JSONDecodeError:
                logging.warning(f"Failed to parse JSON-LD for {dataset_url}")

        # --- Strategy 2: HTML Parsing (Fallback) ---

        # Extract 'URL Estável' (Stable URL)
        dt_tags = soup.find_all("dt")
        for dt in dt_tags:
            if "URL Estável" in dt.get_text(strip=True) or "Stable URL" in dt.get_text(strip=True):
                dd = dt.find_next_sibling("dd")
                if dd:
                    link = dd.find("a")
                    if link and link.get("href"):
                        details["stable_url"] = link.get("href")
                    else:
                        details["stable_url"] = dd.get_text(strip=True)
                break

        # Extract Description
        desc_div = soup.find("div", class_="markdown")
        if desc_div:
            details["full_description"] = desc_div.get_text(strip=True)

        # Extract Metadata (Update Date and Format) from subheaders-infos
        if details["last_updated"] == "N/A" or details["file_formats"] == "N/A":
            metadata_div = soup.find("div", class_="subheaders-infos")
            if metadata_div:
                for span in metadata_div.find_all("span"):
                    text = span.get_text(strip=True)
                    if "Actualizado na" in text and details["last_updated"] == "N/A":
                        details["last_updated"] = text.replace("Actualizado na", "").strip()
                    elif "Formato" in text and details["file_formats"] == "N/A":
                        fmt = text.replace("Formato", "").strip()
                        if fmt:
                            details["file_formats"] = fmt

        # Extract Metadata from fr-text classes (New structure found in debug)
        if details["last_updated"] == "N/A":
            # Look for "Actualizado à"
            update_p = soup.find(lambda tag: tag.name == "p" and "Actualizado à" in tag.get_text())
            if update_p:
                details["last_updated"] = update_p.get_text(strip=True).replace("Actualizado à", "").strip()

        # Extract File Formats (Fallback to buttons if not found in metadata)
        if details["file_formats"] == "N/A":
            formats = set()
            download_links = soup.find_all("a", class_="matomo_download")
            for link in download_links:
                text = link.get_text(strip=True)
                clean_format = text.replace("Descarregar ficheiro como", "").replace("Download file as", "").strip()
                if clean_format:
                    formats.add(clean_format)

            if formats:
                details["file_formats"] = ", ".join(sorted(formats))

        return details

    def process_search_page(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Parses a search result page and processes each dataset found.

        Args:
            soup (BeautifulSoup): The parsed HTML of the search result page.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing dataset info.
        """
        results = []

        # Datasets are usually contained within <article> tags in the search results
        articles = soup.find_all("article", class_="fr-enlarge-link")

        for article in tqdm(articles, desc="Processing datasets", leave=False):
            try:
                # Extract Title
                title_tag = article.find("h4")
                if not title_tag:
                    continue

                title = title_tag.get_text(strip=True)

                # Filter: Skip datasets with specif words in the title
                title_lower = title.lower()

                # List of words to skip ("Resultados" or "Desafio" or "Recenseamento" or
                #                        "População Residente" or "População Presente" or "População Empregada" or
                #                        "Alojamentos Familiares" "Edifícios por" or "Núcleos familiares" or
                #                        "Famílias" or "Agregados familiares" or "Anomalia da"
                #                        "Taxa de desemprego" or "Taxa de atividade" or "Taxa de emprego" or
                #                        "Modelo" in title)
                excluded_words = ["resultados", "desafio", "recenseamento", "população", "alojamentos familiares",
                                  "edifícios por", "núcleos familiares", "famílias", "agregados familiares",
                                  "anomalia da", "taxa de desemprego", "taxa de atividade", "taxa de emprego",
                                  "modelo"]

                if any(word in title_lower for word in excluded_words):
                    self.skipped_count += 1
                    # For debugging purposes
                    # logging.info(f"Skipping excluded dataset: {title}")
                    continue

                # Extract Relative Link
                link_tag = title_tag.find("a")
                if not link_tag:
                    continue

                relative_url = link_tag.get("href")
                full_url = urljoin(self.base_url, relative_url)

                # For debugging purposes
                # logging.info(f"Processing dataset: {title}")

                # Go to detail page to get the Stable URL
                details = self.extract_dataset_details(full_url)

                dataset_info = {
                    "title": title,
                    "url_portal": full_url,
                    "stable_url": details["stable_url"],
                    "description": details["full_description"],
                    "file_formats": details.get("file_formats", "N/A"),
                    "last_updated": details.get("last_updated", "N/A")
                }

                results.append(dataset_info)

                # Polite delay between inner requests
                time.sleep(1)

            except Exception as e:
                logging.error(f"Error processing an article: {e}")
                continue

        return results

    def run(self) -> List[Dict[str, Any]]:
        """
        Main execution method. Iterates through pagination until no more pages are found.

        Returns:
            List[Dict[str, Any]]: A complete list of all scraped datasets.
        """
        all_datasets = []
        current_url = self.search_url
        page_num = 1

        logging.info("Starting extraction of Lisbon Open Data...")

        while current_url:
            logging.info(f"Scraping Search Page {page_num}: {current_url}")

            soup = self._get_soup(current_url)
            if not soup:
                break

            datasets_on_page = self.process_search_page(soup)
            all_datasets.extend(datasets_on_page)

            # Pagination handling
            # Look for the "Next" button in the pagination list
            next_page_link = soup.find("a", class_="fr-pagination__link--next")

            if next_page_link and next_page_link.get("href"):
                next_relative = next_page_link.get("href")
                # Handle cases where href is just query params or full path
                if next_relative.startswith("?"):
                    # Reconstruct URL if it's just query params (common in some frameworks)
                    # However, dados.gov.pt usually provides relative paths
                    current_url = f"{self.base_url}/pt/datasets/{next_relative}"
                else:
                    current_url = urljoin(self.base_url, next_relative)

                page_num += 1
                time.sleep(1)  # Delay between search pages
            else:
                logging.info("No next page found. Finishing scraping.")
                current_url = None

        logging.info(f"Extraction complete. Found {len(all_datasets)} eligible datasets.")
        return all_datasets


def save_to_json(data: List[Dict[str, Any]], filename: str):
    """
    Saves the extracted data to a JSON file.

    Args:
        data (List[Dict[str, Any]]): The data to save.
        filename (str): The output filename.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Data successfully saved to {filename}")
    except IOError as e:
        logging.error(f"Error saving file: {e}")


def main():
    """
    Main function to execute the scraper.
    """
    scraper = LisbonOpenDataScraper()
    data = scraper.run()

    if data:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_filepath = os.path.join(script_dir, 'lisbon_datasets.json')

        # Save to JSON
        save_to_json(data, output_filepath)

        # Print report
        print("\n\033[1m--- Extraction Report ---\033[0m")
        print(f"\033[1m  - Collected:\033[0m {len(data)}")
        print(f"\033[1m  - Skipped:\033[0m {scraper.skipped_count}")
        print(f"\033[1m  - Total processed:\033[0m {len(data) + scraper.skipped_count}")
        print("Extraction complete.")

        # Print a preview
        print("\n\033[1m--- Extraction Preview ---\033[0m")
        for i, item in enumerate(data[:3]):
            print(f"[{i+1}] {item['title']}")
            print(f"\033[1m    Stable URL:\033[0m {item['stable_url']}")
            print(f"\033[1m    Formats:\033[0m {item.get('file_formats', 'N/A')}")
            print(f"\033[1m    Updated:\033[0m {item.get('last_updated', 'N/A')}")
            print("-" * 40)


if __name__ == "__main__":
    main()
