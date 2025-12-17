# GitHub Copilot Instructions for Thesis2025-26_AFGS

This repository contains the code and resources for a Master's Thesis in Data Science and Advanced Analytics.

**Title**: *LLM-Powered Urban Exploration: A Framework for Adaptive Tourist and Mobility Itinerary Planning*
**Author**: André Filipe Gomes Silvestre (20240502)

## 🏗 Project Architecture & Context

- **Goal**: Develop a **Multi-Agent System (MAS)** (not just RAG) that creates personalized tourist itineraries and assists in urban mobility in Lisbon (AML).
- **Core Logic**: The system combines static tourist data with real-time APIs to generate adaptive plans.
- **Key Features**:
  - **Automatic Planning**: Based on preferences, time, weather, and traffic.
  - **Simulation**: Ticket purchasing for transport and attractions. (No real transactions.)
  - **Export**: Integration with Google/Apple Calendar or Notion. (In future work.)

## 📂 Data Sources & APIs

The system relies on the following specific data sources. **Use these endpoints for any API implementation.**

1.  **Tourist Spots**: `Visitlisboa.com`, `Turismo de Lisboa` (PDFs).
2.  **Meteorology (IPMA)**:
    -   Warnings: `https://api.ipma.pt/open-data/forecast/warnings/warnings_www.json`
    -   Daily Forecast: `https://api.ipma.pt/open-data/forecast/meteorology/cities/daily/{globalIdLocal}.json` (Lisbon ID: `1110600`)
3.  **Public Transport**:
    -   **Metro de Lisboa**: `https://app.metrolisboa.pt/status/getLinhas.php`
    -   **Carris Metropolitana**: `https://api.carrismetropolitana.pt/v2/alerts`, `/stops`, `/lines`, `/routes`
    -   **Comboios de Portugal (CP)**: `https://comboios.live/api/stations`, `/vehicles`
4.  **Essential Services**: `Lisboa Aberta CM` (GeoJSON).

## 📝 Coding Conventions

### Python Scripts
- **Header**: Every script must start with this exact header:
  ```python
  # ==========================================================================
  # Master Thesis
  #   - André Filipe Gomes Silvestre, 20240502
  # 
  # [Description of the script]
  # 
  # [Optional: Link to site/repo/API docs]
  # ==========================================================================
  ```
- **Docstrings**: Use the following format for all functions/classes:
  ```python
  def function_name(param1: type) -> return_type:
      """
      Brief description.

      Args:
          param1 (type): Description.

      Returns:
          return_type: Description.

      Notes:
          - Additional notes.
      """
  ```
- **Error Handling**:
  -   Handle HTTP errors (404, 500), timeouts, and connection errors.
  -   **MUST** implement retries with exponential backoff for transient errors.
  -   **MUST** use timeouts for all requests.
- **Anti-Bot**: Use headless browsers (Selenium/others) with stealth plugins to avoid detection.

### Jupyter Notebooks
- **Structure**: Imports -> Config -> Logic.
- **Markdown**: Use clear headings and explanations for every step.

## 🗣️ Tone & Style (Thesis Context)

- **Academic Tone**: Direct, helpful, concise. No over-explaining.
- **Language**: English (default). If Portuguese is requested, **NO Brazilianisms**.
- **Prohibited**: **NEVER** use em-dashes or en-dashes ("—"). Use commas, colons, or parentheses instead. Hyphens ("-") are allowed for compound words.
- **Citations**: Use **APA 7th edition** style.
- **Criticism**: Push back if a request contradicts best practices or facts.

## 📦 Dependencies

- **Environment**: Keep `requirements.txt` updated.
