# Project Documentation

**Master Thesis - LLM-Powered Urban Exploration**  
**Author:** André Filipe Gomes Silvestre (20240502)  
**Institution:** NOVA IMS - Master in Data Science and Advanced Analytics  
**Year:** 2025-2026

---

## 📚 Documentation Structure

This documentation provides comprehensive information about the Lisbon Urban Assistant project, including code architecture, API references, data schemas, and implementation details.

### Directory Structure

```
docs/
├── README.md                          # This file
├── api/                               # API Documentation
│   ├── tools_overview.md              # Overview of all tools
│   ├── ipma_api.md                    # Weather API documentation
│   ├── transport_api.md               # Transport API documentation
│   ├── dados_abertos.md               # Open Data API documentation
│   ├── visitlisboa_api.md             # VisitLisboa semantic search documentation
│   └── vector_store.md                # Vector database documentation
├── architecture/                      # System Architecture
│   ├── system_overview.md             # High-level system architecture
│   ├── agent_architecture.md          # LangGraph agent design
│   ├── data_flow.md                   # Data flow diagrams and explanations
│   └── technology_stack.md            # Technologies and dependencies
├── datasets/                          # Dataset Documentation
│   ├── visitlisboa_events.md          # Events dataset schema and examples
│   ├── visitlisboa_places.md          # Places dataset schema and examples
│   ├── lisbon_open_data.md            # Dados Abertos datasets documentation
│   └── vector_database.md             # ChromaDB collections documentation
└── processes/                         # Process Documentation
    ├── webscraping.md                 # Web scraping implementation
    ├── vector_sync.md                 # Vector store synchronization
    ├── error_handling.md              # Error handling strategies
    └── deployment.md                  # Deployment and GitHub Actions
```

---

## 🎯 Quick Start

### For Developers

1. **Architecture Overview**: Start with [system_overview.md](architecture/system_overview.md)
2. **Agent Design**: Read [agent_architecture.md](architecture/agent_architecture.md)
3. **Available Tools**: Check [tools_overview.md](api/tools_overview.md)

### For Data Scientists

1. **Datasets**: Explore [datasets/](datasets/) folder
2. **Vector Store**: Read [vector_store.md](api/vector_store.md)
3. **Data Flow**: See [data_flow.md](architecture/data_flow.md)

### For Researchers

1. **System Architecture**: [system_overview.md](architecture/system_overview.md)
2. **Technology Stack**: [technology_stack.md](architecture/technology_stack.md)
3. **Implementation Details**: All files in [processes/](processes/)

---

## 🔑 Key Components

### 1. Multi-Agent System
- **Supervisor Agent**: Routes queries to specialized agents
- **Weather Agent**: IPMA weather data and forecasts
- **Transport Agent**: Metro, bus, tram, and train information
- **Researcher Agent**: RAG for places and events
- **Planner Agent**: Itinerary synthesis
- **40 specialized tools** for different data sources
- **Multi-provider**: Supports LM Studio (default), Groq, Google, OpenAI, Ollama

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

## 📖 Documentation Conventions

### Code Examples
All code examples are tested and working. They include:
- **Type hints** for clarity
- **Docstrings** in Google style
- **Error handling** examples
- **Real-world usage** scenarios

### Diagrams
- **ASCII diagrams** for terminal readability
- **Mermaid diagrams** for visual representation
- **Flow charts** for process understanding

### API Documentation
- **Function signature** with types
- **Parameters** with descriptions
- **Return values** with type information
- **Examples** with expected output
- **Error cases** and handling

---

## 🔬 Research Context

This project is part of a Master's thesis exploring:
- **LLM-powered urban navigation** in smart cities
- **Multi-agent systems** for tourism and mobility
- **RAG (Retrieval-Augmented Generation)** for local knowledge
- **Real-time data integration** with LLMs
- **Adaptive itinerary planning** based on conditions

---

## 📊 Statistics

### Code Base
- **Python Modules**: 25+
- **Tools Implemented**: 40
- **Specialized Agents**: 5
- **Data Sources**: 6 APIs + 2 scraped sources
- **Vector DB Documents**: ~1,400 chunks

### Datasets
- **Events**: 200+ cultural events
- **Places**: 300+ attractions and services
- **Open Data**: 100+ GeoJSON datasets
- **PDF Guide**: 100+ pages indexed

---

## 🚀 Getting Started

For a complete project setup, see the main [README.md](../README.md) in the project root.

For specific documentation, navigate to the appropriate section above.

---

## 📝 Contributing to Documentation

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

*Last Updated: January 28, 2026*
