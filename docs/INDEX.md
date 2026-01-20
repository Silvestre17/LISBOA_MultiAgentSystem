# Documentation Index & Quick Reference

**Project**: LLM-Powered Urban Exploration Framework  
**Author**: André Filipe Gomes Silvestre (20240502)  
**Institution**: NOVA IMS - Master in Data Science and Advanced Analytics  
**Year**: 2025-2026

---

## 📚 Documentation Files

### Main Documentation
1. **[README.md](README.md)** - Documentation overview and structure
2. **[COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md)** - **⭐ START HERE** - Complete technical reference (100+ pages)

### Architecture
3. **[ARCHITECTURE_DIAGRAMS.md](architecture/ARCHITECTURE_DIAGRAMS.md)** - Visual system architecture and workflows

### API Reference
4. **[tools_overview.md](tools_overview.md)** - Overview of all 29 tools with usage guide

---

## 🚀 Quick Start Guide

### For Developers

**1. Understanding the System**:
- Read: [ARCHITECTURE_DIAGRAMS.md](architecture/ARCHITECTURE_DIAGRAMS.md) sections 1-2
- Understand: Agent flow and tool execution

**2. Exploring the Code**:
- **Agent**: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#2-agent-components)
- **Tools**: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#3-tools-api-reference)
- **State**: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#22-state-management)

**3. Running Tests**:
```bash
# Test individual tools
python tools/ipma_api.py
python tools/transport_api.py
python tools/visitlisboa_api.py

# Test agent
python agent/graph.py

# Validate vector store
python tools/vector_store.py --test
```

---

### For Data Scientists

**1. Dataset Exploration**:
- **Events**: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#41-visitlisboa-events)
- **Places**: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#42-visitlisboa-places)
- **Open Data**: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#43-lisboa-aberta-open-data)

**2. Vector Store**:
- **Architecture**: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#35-vector-store)
- **Sync Process**: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#52-vector-store-synchronization)

**3. Real Data Examples**:
```python
# Load events dataset
import json
with open('data_collection/webscraping/events.json') as f:
    events = json.load(f)
print(f"Total events: {len(events)}")
print(f"First event: {events[0]}")
```

---

### For Researchers

**1. System Design**:
- **Overview**: [ARCHITECTURE_DIAGRAMS.md](architecture/ARCHITECTURE_DIAGRAMS.md#1-high-level-system-architecture)
- **Agent Pattern**: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#21-agent-architecture)
- **Data Flow**: [ARCHITECTURE_DIAGRAMS.md](architecture/ARCHITECTURE_DIAGRAMS.md#5-data-sources-integration)

**2. Implementation Details**:
- **Web Scraping**: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#51-web-scraping)
- **Error Handling**: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#6-error-handling)
- **Testing**: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#7-testing--validation)

**3. Automation**:
- **GitHub Actions**: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#53-github-actions-automation)

---

## 📊 Project Statistics

### Code Base
- **Python Modules**: 20+
- **Lines of Code**: ~8,000+
- **Tools Implemented**: 29
- **Specialized Agents**: 5
- **Data Sources**: 4 APIs + 2 scraped sources

### Datasets
- **Events**: 200+ cultural events
- **Places**: 300+ attractions
- **Open Data**: 100+ GeoJSON datasets
- **PDF Guide**: ~900 text chunks

### Vector Store
- **Total Documents**: ~1,400
- **Embedding Model**: BAAI/bge-m3 (multilingual)
- **Database**: ChromaDB (persistent)
- **Collections**: 3 (PDF, places, events)

---

## 🔍 Finding Information

### By Topic

**Weather Data**:
- API: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#31-weather-tools-ipma)
- Example: See "get_weather_forecast()" section

**Transport Data**:
- APIs: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#32-transport-tools)
- Flow: [ARCHITECTURE_DIAGRAMS.md](architecture/ARCHITECTURE_DIAGRAMS.md#3-tool-execution-workflow)

**Events & Places**:
- Search: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#34-visitlisboa-tools)
- Datasets: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#4-dataset-documentation)

**Agent Behavior**:
- Architecture: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#21-agent-architecture)
- Prompts: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#23-system-prompts)
- State: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#22-state-management)

**LLM Integration**:
- Factory Pattern: [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#24-llm-factory)
- Providers: See "Supported Providers" section

---

## 🛠️ Common Tasks

### Task 1: Add a New Tool

**Documentation to Read**:
1. [tools_overview.md](api/tools_overview.md) - Understand tool structure
2. [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#3-tools-api-reference) - See examples

**Steps**:
```python
# 1. Create tool in tools/your_tool.py
from langchain_core.tools import tool

@tool
def your_tool(param: str) -> str:
    """Tool description."""
    # Implementation
    return result

# 2. Register in tools/__init__.py
from tools.your_tool import your_tool
__all__ = [..., "your_tool"]

# 3. Add to agent/graph.py
from tools.your_tool import your_tool
# Add to get_all_tools() list
```

---

### Task 2: Modify Agent Behavior

**Documentation to Read**:
1. [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#23-system-prompts)
2. [ARCHITECTURE_DIAGRAMS.md](architecture/ARCHITECTURE_DIAGRAMS.md#2-langgraph-agent-flow)

**Files to Edit**:
- `agent/prompts.py` - System prompt
- `agent/state.py` - State schema (if adding context)
- `agent/graph.py` - Graph structure (if changing flow)

---

### Task 3: Update Datasets

**Documentation to Read**:
1. [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#51-web-scraping)
2. [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#52-vector-store-synchronization)

**Manual Update**:
```bash
# Scrape new data
python data_collection/webscraping/visitlisbon_events.py
python data_collection/webscraping/visitlisbon_places.py

# Sync vector store
python tools/vector_store.py
```

**Automated** (already configured):
- GitHub Actions runs daily at 2 AM UTC (scraping)
- GitHub Actions runs daily at 3 AM UTC (vector sync)

---

### Task 4: Test a Specific Component

**Documentation to Read**:
1. [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#7-testing--validation)

**Test Commands**:
```bash
# Weather tools
python tools/ipma_api.py

# Transport tools
python tools/transport_api.py

# Dados Abertos (Open Data)
python tools/dados_abertos.py

# VisitLisboa (semantic search)
python tools/visitlisboa_api.py

# Vector store
python tools/vector_store.py --test

# Agent
python agent/graph.py

# LLM Factory
python agent/llm_factory.py
```

---

### Task 5: Deploy the System

**Documentation to Read**:
1. Main [../README.md](../README.md) - Setup instructions
2. [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#53-github-actions-automation)
3. [ARCHITECTURE_DIAGRAMS.md](architecture/ARCHITECTURE_DIAGRAMS.md#8-deployment-architecture-github-actions)

**Local Deployment**:
```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys (.env file)
GROQ_API_KEY=your_key_here

# Build vector store (first time only)
python tools/vector_store.py

# Run Streamlit app
streamlit run app.py
```

**Cloud Deployment** (Streamlit Cloud):
1. Push to GitHub
2. Connect Streamlit Cloud to repository
3. Set environment variables (API keys)
4. Deploy

---

## 📝 Code Examples

### Example 1: Using a Tool Directly

```python
from tools.ipma_api import get_weather_forecast

# Get 5-day forecast
forecast = get_weather_forecast.invoke({"days": 5})
print(forecast)
```

### Example 2: Querying the Agent

```python
from agent.graph import create_assistant

# Create assistant
assistant = create_assistant()

# Ask question
response = assistant.chat("What's the weather in Lisbon?")
print(response)
```

### Example 3: Searching Events

```python
from tools.visitlisboa_api import search_cultural_events

# Search events this weekend
events = search_cultural_events.invoke({
    "date_filter": "this weekend",
    "max_results": 10
})
print(events)
```

### Example 4: Finding Nearby Services

```python
from tools.dados_abertos import find_nearby_services

# Find pharmacies near user
services = find_nearby_services.invoke({
    "service_type": "farmácias",
    "user_lat": 38.7223,
    "user_lon": -9.1393,
    "max_results": 5
})
print(services)
```

---

## ⚠️ Important Notes

### API Keys Required
- **Groq**: Free tier (14,400 requests/day)
- **Google**: Free tier (60 requests/minute)
- **OpenAI**: Pay-per-use

Get keys from:
- Groq: https://console.groq.com
- Google: https://aistudio.google.com/apikey
- OpenAI: https://platform.openai.com/api-keys

### Environment Setup

**`.env` file**:
```bash
# LLM Provider Keys
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...
OPENAI_API_KEY=sk-...

# LangSmith (Optional - for tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=thesis-project
```

### Common Issues

**Issue**: "Vector store unavailable"
- **Solution**: Run `python tools/vector_store.py` to build database

**Issue**: "API timeout"
- **Solution**: Check internet connection, APIs may be temporarily down

**Issue**: "No results found"
- **Solution**: Try broader search terms, check date filters

---

## 📧 Support & Contact

**Author**: André Filipe Gomes Silvestre  
**Student ID**: 20240502  
**Institution**: NOVA IMS  
**Program**: Master in Data Science and Advanced Analytics  

---

## 🔄 Documentation Updates

### Version History
- **v2.0** (Jan 20, 2026): Multi-Agent System update
  - Added 5 specialized agents (Supervisor, Weather, Transport, Researcher, Planner)
  - Expanded from 18 to 29 tools
  - Added Metro Official API with OAuth2
  - Renamed Carris functions to Carris Metropolitana
  - Added CP AML stations
- **v1.0** (Dec 30, 2025): Initial comprehensive documentation
  - Complete API reference
  - Architecture diagrams
  - Dataset documentation
  - Process workflows

### Future Updates
- Additional tool documentation as tools are added
- Performance benchmarks
- User study results
- Deployment case studies

---

## 📚 External Resources

### LangGraph
- Official Docs: https://langchain-ai.github.io/langgraph/
- ReAct Pattern: https://www.promptingguide.ai/techniques/react

### ChromaDB
- Official Docs: https://docs.trychroma.com/
- Embeddings Guide: https://www.trychroma.com/embeddings

### APIs
- IPMA: https://api.ipma.pt/
- Metro Lisboa: https://www.metrolisboa.pt/
- Carris Metropolitana: https://api.carrismetropolitana.pt/
- Dados Abertos: https://dados.gov.pt/

---

*This index serves as the central navigation point for all project documentation.*

**Last Updated**: January 20, 2026
