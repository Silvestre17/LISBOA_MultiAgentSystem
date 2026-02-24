# Project Documentation

**Master Thesis - LISBOA: Lisbon Itenerary System Based On AI**  
**Subtitle: A Multi-Agent Approach for Personalized Tourism and Urban Mobility in Lisbon**  
**Author:** André Filipe Gomes Silvestre (20240502)  
**Institution:** NOVA IMS - Master in Data Science and Advanced Analytics  
**Year:** 2025-2026

---

## Documentation Structure

This documentation provides comprehensive information about the Lisbon Urban Assistant project, including code architecture, API references, data schemas, and implementation details.

The documentation is now being consolidated into a flat set of grouped Markdown files:

- `docs/00_INDEX.md` (entrypoint)
- `docs/01_PROJECT_OVERVIEW.md`
- `docs/02_SYSTEM_ARCHITECTURE.md`
- `docs/03_TOOLS_REFERENCE.md`
- `docs/04_DATA_SOURCES_AND_SCHEMAS.md`
- `docs/05_DEPLOYMENT_AND_OPERATIONS.md`
- `docs/06_FUTURE_ENHANCEMENTS.md`

Legacy documents remain during migration to avoid information loss.

---

## Quick start

Start here: `docs/00_INDEX.md`.

---

## Key components

### 1. Multi-Agent System
- **Supervisor Agent**: Routes queries to specialized agents
- **Weather Agent**: IPMA weather data and forecasts
- **Transport Agent**: Metro, bus, tram, and train information
- **Researcher Agent**: RAG for places and events
- **Planner Agent**: Itinerary synthesis
- **42 specialized tools** for different data sources
- **Multi-provider**: Supports LM Studio (default), OpenAI, Azure

### 2. Data Sources
- **IPMA**: Weather forecasts and warnings
- **Metro de Lisboa**: Official API - Line status, wait times, frequencies
- **Carris Urban**: GTFS data for city buses and trams (28E, 15E, 732...)
- **Carris Metropolitana**: Suburban bus alerts, stops, routes, real-time tracking
- **CP (Comboios de Portugal)**: Train status, delays, AML stations
- **Lisboa Aberta**: Open government data (GeoJSON)
- **VisitLisboa**: Cultural events and tourist attractions

### 3. Vector Store (RAG)
- **ChromaDB**: Persistent vector database
- **Embeddings**: BAAI/bge-m3 multilingual model
- **Collections**: PDF guide, places, events
- **Incremental Sync**: Efficient updates

### 4. Web Scraping
- **VisitLisboa Events**: Automated daily scraping
- **VisitLisboa Places**: Automated weekly scraping
- **Dados.gov**: Metadata extraction for open datasets

---

## Documentation conventions

- Documentation is English-only.
- The root docs are flat, grouped by topic.
- Examples aim to be runnable, but should be treated as illustrative unless explicitly marked as tested.

---

## Research context

This project is part of a Master's thesis exploring:
- **Multi-agent orchestration** for urban navigation and tourism in smart cities
- **Multi-agent systems** for tourism and mobility
- **RAG (Retrieval-Augmented Generation)** for local knowledge
- **Real-time data integration** with LLMs
- **Adaptive itinerary planning** based on conditions

---

## Statistics

### Code Base
- **Python Modules**: 25+
- **Tools Implemented**: 42
- **Specialized Agents**: 5
- **Data Sources**: 6 APIs + 2 scraped sources
- **Vector DB Documents**: ~1,400 chunks

### Datasets
- **Events**: 200+ cultural events
- **Places**: 300+ attractions and services
- **Open Data**: 100+ GeoJSON datasets
- **PDF Guide**: 100+ pages indexed

---

## Getting started

For a complete project setup, see the main [README.md](../README.md) in the project root.

For specific documentation, navigate to the appropriate section above.

---

## Contributing to documentation

When updating code:
1. Update corresponding API documentation
2. Add examples if behavior changes
3. Update architecture diagrams if structure changes
4. Document error handling strategies

---

## 📧 Contact

**André Filipe Gomes Silvestre**  
Student ID: 20240502  
NOVA IMS - Master in Data Science and Advanced Analytics

---

*Last Updated: February 4, 2026*
