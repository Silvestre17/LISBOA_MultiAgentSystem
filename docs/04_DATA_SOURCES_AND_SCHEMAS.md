# Data collections

This document describes how datasets are collected, stored, and updated.

## VisitLisboa scraping outputs

Scripts:

- Events scraper: `data_collection/webscraping/visitlisbon_events.py`
- Places scraper: `data_collection/webscraping/visitlisbon_places.py`

Outputs:

- `data_collection/webscraping/events.json`
- `data_collection/webscraping/places.json`

Incremental update behavior:

- Both scrapers implement delta logic: they load the existing JSON, scrape the current URL list, add new items, update changed items, and keep unchanged items.
- If the scraper finds zero URLs, it aborts without overwriting the local JSON (safety guard).

### events.json schema (high level)

Each event is a dictionary with fields such as:

- `url`
- `title`
- `category`
- `short_description`
- `full_description`
- `image_urls` (list)
- `video_urls` (list)
- `dates` (list)
- `price` (optional)
- `venue_name` (optional)
- `location` (optional)
- `buy_tickets_url` (optional)
- `information_links` (dict of label to URL)

Dates are normalized as a list of entries. Examples:

- Single date: `{ "type": "single", "date": {"datetime_iso": "YYYY-MM-DD", "display_text": "...", "time": "HH:MM" or null} }`
- Range: `{ "type": "range", "start": {...}, "end": {...} }`

**Complete Example:**

```json
{
  "url": "https://www.visitlisboa.com/pt-pt/eventos/festa-santo-antonio-2026",
  "title": "Festa de Santo António 2026",
  "category": "Festas e Romarias",
  "short_description": "Celebração anual do Santo Padroeiro de Lisboa",
  "full_description": "A Festa de Santo António é a maior celebração popular de Lisboa, com marchas, sardinhas assadas, e arraiais por toda a cidade. As festividades começam na véspera (12 de junho) e prolongam-se pela madrugada do dia 13.",
  "image_urls": [
    "https://www.visitlisboa.com/sites/default/files/santo_antonio_2026.jpg"
  ],
  "video_urls": [],
  "dates": [
    {
      "type": "range",
      "start": {
        "datetime_iso": "2026-06-12",
        "display_text": "12 Jun 2026",
        "time": "21:00"
      },
      "end": {
        "datetime_iso": "2026-06-13",
        "display_text": "13 Jun 2026",
        "time": "02:00"
      }
    }
  ],
  "price": "Gratuito",
  "venue_name": "Bairro de Alfama",
  "location": "Alfama, Lisboa",
  "buy_tickets_url": null,
  "information_links": {
    "Programa Oficial": "https://www.visitlisboa.com/festas-lisboa"
  }
}
```

### places.json schema (high level)

Each place is a dictionary with fields such as:

- `url`
- `title`
- `category`
- `short_description`
- `full_description`
- `image_urls` (list)
- `video_urls` (list)
- `features` (list)
- `location` (optional)
- `contact_info` (dict, may include `phone`, `email`, `website`, `tickets_url`)
- `social_media` (dict)
- `schedules` (list of schedule entries)
- `tickets_offers` (optional)
- `tripadvisor` (optional)

Schedules support multiple blocks (for example summer and winter schedules), and may include `today`, a `date_range`, and an `hours` dictionary.

Local run:

```bash
python data_collection/webscraping/visitlisbon_events.py
python data_collection/webscraping/visitlisbon_places.py
```

## Lisboa Aberta metadata

- `data_collection/webscraping/lisbon_datasets_clean.json`

This metadata file is used by the Lisboa Aberta tools to:

- list datasets by category or keyword
- fetch GeoJSON on demand
- inspect dataset fields and sample records

## Transport reference data

Carris Urban:

- `data/carris/carris.db`
- `data/carris/metadata.json`

CP:

- `data/cp/metadata.json`

## Vector database

Persistent Chroma directory:

- `data/vector_db/`

Collections:

- `lisbon_pdf`
- `lisbon_places`
- `lisbon_events`

Build or update:

```bash
python tools/vector_store.py
```
