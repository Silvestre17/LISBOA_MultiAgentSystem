<p align="center">
   <a href="https://github.com/Silvestre17/Thesis2025-26_AFGS">
        <img src="./docs/img/thesis_banner.png" alt="Thesis Project Banner" width="800">
    </a>
</p>

# 🗺️ LLM-Powered Urban Exploration: Adaptive Tourist & Mobility Itinerary Planning 🤖

<p align="center">
    <!-- Project Repository and App Badges -->
    <a href="https://github.com/Silvestre17/Thesis2025-26_AFGS"><img src="https://img.shields.io/badge/Project_Repo-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repo"></a>
    <a href="#"><img src="https://img.shields.io/badge/Streamlit_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit App"></a>
</p>

## 📝 Description

This project develops an **AI-powered Multi-Agent System** for personalized tourist itinerary planning and real-time urban mobility assistance in **Lisbon** 🇵🇹. Using **LangGraph** and **Retrieval-Augmented Generation (RAG)**, the system intelligently combines static knowledge with real-time data from multiple APIs to create adaptive, context-aware recommendations for visitors and residents exploring the city.

The system acts as an intelligent urban assistant, answering questions about cultural events, tourist attractions, weather conditions, public transport status, and nearby services while maintaining conversational context and providing data-driven, personalized suggestions.

<p align="center">
    <a href="https://www.visitlisboa.com/"><img src="https://img.shields.io/badge/VisitLisboa-FF6B35?style=for-the-badge&logo=tourism&logoColor=white" alt="VisitLisboa" /></a>
    <a href="https://www.ipma.pt/"><img src="https://img.shields.io/badge/IPMA-0052CC?style=for-the-badge&logo=weather&logoColor=white" alt="IPMA Weather" /></a>
    <a href="https://www.metrolisboa.pt/"><img src="https://img.shields.io/badge/Metro_Lisboa-E60000?style=for-the-badge&logo=metro&logoColor=white" alt="Metro de Lisboa" /></a>
    <a href="https://www.carrismetropolitana.pt/"><img src="https://img.shields.io/badge/Carris_Metropolitana-00A859?style=for-the-badge&logo=bus&logoColor=white" alt="Carris Metropolitana" /></a>
</p>

## 🎓 Project Context

This project is the **Master's Thesis** for the **[Master's in Data Science and Advanced Analytics](https://www.novaims.unl.pt/en/education/programs/postgraduate-programs-and-master-degree-programs/master-degree-program-in-data-science-and-advanced-analytics-with-a-specialization-in-data-science/)** program at **NOVA IMS**, developed during the **2025/2026 academic year**.

**Thesis Title:** *LLM-Powered Urban Exploration: A Framework for Adaptive Tourist and Mobility Itinerary Planning*  
**Author:** André Filipe Gomes Silvestre (Student ID: 20240502)  
**Supervisor:** TBD  
**Institution:** NOVA Information Management School (NOVA IMS)

## ✨ Objective

The primary objectives of this thesis are to:

- **Develop a Multi-Agent System** using LangGraph's ReAct pattern for intelligent urban exploration assistance.
- **Integrate Real-Time Data Sources** (weather, transport, events) with static knowledge (tourist guides, POIs).
- **Implement RAG (Retrieval-Augmented Generation)** for semantic search over cultural events, places, and local knowledge.
- **Create an Adaptive Itinerary Planner** that considers user preferences, real-time conditions, and dynamic constraints.
- **Evaluate LLM Performance** across multiple providers (Groq, Google, OpenAI, local models) for conversational AI tasks.
- **Automate Data Collection** through web scraping and GitHub Actions for continuous knowledge base updates.

## 🏗️ System Architecture

The system follows a **modular, tool-based architecture** powered by **LangGraph** for agent orchestration:

<p align="center">
    <img src="./docs/architecture/system_diagram.png" alt="System Architecture" width="800" style="background-color: white;">
</p>
<p align="center"><i><b>Figure 1:</b> High-Level System Architecture (see <a href="./docs/architecture/ARCHITECTURE_DIAGRAMS.md">ARCHITECTURE_DIAGRAMS.md</a> for detailed diagrams).</i></p>

### Core Components

1. **Multi-Agent System (LangGraph)** 🤖
   - **Supervisor Agent**: Orchestrator that analyzes queries and routes them to the most appropriate specialized agent (parallel execution supported)
   - **Weather Agent**: IPMA weather data and forecasts
   - **Transport Agent**: Metro, bus, and train information
   - **Researcher Agent**: RAG for places and events
   - **Planner Agent**: Itinerary synthesis
   - **29 specialized tools** for different data sources
   - **Multi-provider LLM support** (LM Studio, Groq, Google, OpenAI, Ollama)

2. **Vector Store (RAG)** 📚
   - **ChromaDB** with **BAAI/bge-m3** multilingual embeddings
   - **3 collections**: PDF guide (~900 chunks), places (~300), events (~200)
   - **Incremental synchronization** with content hashing for efficient updates
   - **Semantic search** with fallback mechanisms

3. **Real-Time APIs** ⚡
   - **IPMA**: Weather forecasts, warnings, current conditions
   - **Metro de Lisboa**: Official API with OAuth2 - Line status, wait times, frequencies
   - **Carris Metropolitana**: Bus alerts, stops, routes, real-time arrivals, GPS tracking
   - **CP (Comboios de Portugal)**: Train status, delays, AML stations

4. **Static Knowledge** 📊
   - **VisitLisboa**: Cultural events, tourist attractions (web-scraped)
   - **Lisboa Aberta**: 100+ GeoJSON datasets (open government data)
   - **Tourism Guide**: Comprehensive PDF indexed in vector store

5. **Automated Data Pipeline** 🔄
   - **GitHub Actions**: Daily scraping (2 AM UTC), vector sync (3 AM UTC)
   - **Web scrapers** for VisitLisboa events and places
   - **Anti-bot measures**: Random User-Agent, delays, retry logic

<p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
    <a href="https://www.langchain.com/langgraph"><img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph"></a>
    <a href="https://www.langchain.com/"><img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain"></a>
    <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"></a>
</p>

## 🛠️ Technology Stack

### LLM Providers & Models

<p align="center">
    <a href="https://groq.com/"><img src="https://img.shields.io/badge/Groq-000000?style=for-the-badge&logo=groq&logoColor=white" alt="Groq"></a>
    <a href="https://ai.google.dev/"><img src="https://img.shields.io/badge/Google_Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Google Gemini"></a>
    <a href="https://openai.com/"><img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI"></a>
    <a href="https://lmstudio.ai/"><img src="https://img.shields.io/badge/LM_Studio-000000?style=for-the-badge&logo=lmstudio&logoColor=white" alt="LM Studio"></a>
    <a href="https://ollama.ai/"><img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama"></a>
</p>

**Supported Models:**
- **LM Studio** (default): Local models via `qwen/qwen3-4b-2507` or any OpenAI-compatible model
- **Groq**: `llama-3.3-70b-versatile`, `llama-3.1-70b-versatile`, `qwen3-4b-2507`
- **Google**: `gemini-2.0-flash-exp`, `gemini-1.5-flash`, `gemini-1.5-pro`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- **Ollama**: Any locally installed model

### Data & ML Stack

<p align="center">
    <a href="https://www.trychroma.com/"><img src="https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge&logo=chroma&logoColor=white" alt="ChromaDB"></a>
    <a href="https://huggingface.co/BAAI/bge-m3"><img src="https://img.shields.io/badge/BGE--M3-FFC700?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Embeddings"></a>
    <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"></a>
    <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"></a>
</p>

### Web Scraping & Automation

<p align="center">
    <a href="https://www.crummy.com/software/BeautifulSoup/"><img src="https://img.shields.io/badge/BeautifulSoup-59666C?style=for-the-badge&logo=python&logoColor=white" alt="BeautifulSoup"></a>
    <a href="https://requests.readthedocs.io/"><img src="https://img.shields.io/badge/Requests-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Requests"></a>
    <a href="https://github.com/features/actions"><img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white" alt="GitHub Actions"></a>
</p>

## 📊 Data Sources & Integration

### 1. Weather Data (IPMA) ⛅
- **API**: [Instituto Português do Mar e da Atmosfera](https://api.ipma.pt/)
- **Coverage**: Lisbon (Global ID: 1110600)
- **Data**: 5-day forecasts, weather warnings, current conditions
- **Update Frequency**: Real-time API calls

### 2. Transport Data 🚇🚌🚆

<p align="center">
    <a href="https://www.metrolisboa.pt/"><img src="https://img.shields.io/badge/Metro-E60000?style=for-the-badge&logo=metro&logoColor=white" alt="Metro"></a>
    <a href="https://www.carrismetropolitana.pt/"><img src="https://img.shields.io/badge/Carris-00A859?style=for-the-badge&logo=bus&logoColor=white" alt="Carris"></a>
    <a href="https://www.cp.pt/"><img src="https://img.shields.io/badge/CP_Trains-003DA5?style=for-the-badge&logo=train&logoColor=white" alt="CP"></a>
</p>

- **Metro de Lisboa**: Line status (4 lines: Amarela, Azul, Verde, Vermelha)
- **Carris Metropolitana**: Alerts, stops, routes, real-time arrivals
- **CP (via Comboios.live)**: Train status, delays, vehicle tracking
- **Update Frequency**: Real-time API calls

### 3. Tourist Information 🏛️

- **VisitLisboa**: 200+ cultural events, 300+ attractions
  - **Source**: [VisitLisboa.com](https://www.visitlisboa.com/)
  - **Collection Method**: Automated web scraping (daily)
  - **Data**: Titles, descriptions, dates, locations, schedules, images

### 4. Open Government Data 📂

- **Lisboa Aberta**: 100+ GeoJSON datasets
  - **Source**: [dados.cm-lisboa.pt](https://dados.cm-lisboa.pt/)
  - **Data**: Pharmacies, ATMs, public services, cultural spaces
  - **Features**: Proximity search with Haversine distance calculation

## 🚀 Project Workflow

<p align="center">
    <img src="./docs/architecture/workflow_diagram.png" alt="Workflow" width="800" style="background-color: white;">
</p>
<p align="center"><i><b>Figure 2:</b> End-to-End Data Flow and Agent Execution (see <a href="./docs/COMPLETE_DOCUMENTATION.md">COMPLETE_DOCUMENTATION.md</a> for details).</i></p>

### 1. Data Collection & Preparation 🔍

**Automated Daily Pipeline (GitHub Actions):**
- **02:00 UTC**: Scrape VisitLisboa events and places
- **03:00 UTC**: Sync vector store with new data (incremental updates)

**Manual Setup:**
- Index tourism PDF guide into ChromaDB
- Fetch Lisboa Aberta metadata (100+ datasets)

<p align="center">
    <a href="https://www.beautifulsoup.org/"><img src="https://img.shields.io/badge/BeautifulSoup-59666C?style=for-the-badge&logo=python&logoColor=white" alt="BeautifulSoup"></a>
    <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"></a>
</p>

### 2. Vector Store Synchronization 📚

**Incremental Update Process:**
- **Content Hashing** (SHA-256) detects changes
- **Add new documents**, **update modified**, **delete removed**
- **Embedding**: BAAI/bge-m3 (768-dim multilingual)
- **Storage**: ChromaDB (persistent local database)

<p align="center">
    <a href="https://www.trychroma.com/"><img src="https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge&logo=chroma&logoColor=white" alt="ChromaDB"></a>
    <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/HuggingFace-FFC700?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace"></a>
</p>

### 3. Agent Execution (LangGraph ReAct) 🤖

**User Query** → **Agent Node** (LLM reasoning) → **Tool Selection** → **Tool Execution** → **Result Synthesis** → **Response**

**Key Features:**
- **Persistent State**: Conversation history, user context, itinerary plans
- **Tool Chaining**: Agent can call multiple tools sequentially or in parallel
- **Error Handling**: Retry logic, fallback mechanisms, graceful degradation
- **Dynamic Prompts**: Date/time injection, context-aware instructions

<p align="center">
    <a href="https://www.langchain.com/langgraph"><img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph"></a>
    <a href="https://www.langchain.com/"><img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain"></a>
</p>

### 4. Deployment (Streamlit) 🌐

**Interactive Chat Interface:**
- **Real-time streaming** of agent responses
- **Conversation history** with session persistence
- **LLM provider selection** (Groq, Google, OpenAI, local)
- **Model configuration** UI

<p align="center">
    <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"></a>
</p>

## 📂 Repository Structure

```
Thesis2025-26_AFGS/
├── agent/                              # LangGraph Agent Components
│   ├── graph.py                        # Agent graph definition (ReAct pattern)
│   ├── state.py                        # State schema (TypedDict)
│   ├── prompts.py                      # System prompts and instructions
│   ├── llm_factory.py                  # LLM provider factory (5 providers)
│   └── __init__.py
│
├── tools/                              # Agent Tools (27 total)
│   ├── ipma_api.py                     # Weather tools (3 tools)
│   ├── transport_api.py                # Transport tools (16 tools)
│   ├── dados_abertos.py                # Open data tools (3 tools)
│   ├── visitlisboa_api.py              # VisitLisboa search tools (5 tools)
│   ├── vector_store.py                 # RAG tools (sync + search)
│   └── __init__.py
│
├── data_collection/                    # Data Collection Scripts
│   ├── webscraping/
│   │   ├── visitlisbon_events.py       # Event scraper
│   │   ├── visitlisbon_places.py       # Places scraper
│   │   ├── dadosabertos.gov_lisboa.py  # Open data metadata scraper
│   │   ├── events.json                 # Scraped events (200+)
│   │   └── places.json                 # Scraped places (300+)
│   └── APIs/
│       └── API_IPMA_Metro_Carris_CP.ipynb  # API testing notebook
│
├── data/                               # Data Storage
│   └── vector_db/                      # ChromaDB persistent storage
│       ├── chroma.sqlite3              # Database file
│       └── [collection_folders]/       # Embedding collections
│
├── docs/                               # 📚 COMPREHENSIVE DOCUMENTATION
│   ├── INDEX.md                        # Quick reference & navigation
│   ├── README.md                       # Documentation overview
│   ├── COMPLETE_DOCUMENTATION.md       # ⭐ Full technical reference (100+ pages)
│   ├── architecture/
│   │   └── ARCHITECTURE_DIAGRAMS.md    # 8 ASCII diagrams
│   └── api/
│       └── tools_overview.md           # All 18 tools reference
│
├── .github/
│   └── workflows/
│       ├── daily_scraping.yml          # Daily 2 AM UTC scraping
│       └── update_vector_db.yml        # Daily 3 AM UTC vector sync
│
├── app.py                              # Streamlit application
├── config.py                           # Configuration (API endpoints, IDs)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── LICENSE
```

## 📚 Documentation

**Complete documentation** is available in the [`docs/`](./docs/) folder:

- **[INDEX.md](./docs/INDEX.md)** - Quick reference guide with navigation
- **[COMPLETE_DOCUMENTATION.md](./docs/COMPLETE_DOCUMENTATION.md)** - ⭐ **START HERE** - Full technical reference (100+ pages)
- **[ARCHITECTURE_DIAGRAMS.md](./docs/architecture/ARCHITECTURE_DIAGRAMS.md)** - 8 detailed system diagrams
- **[tools_overview.md](./docs/tools_overview.md)** - API reference for all 29 tools

### Quick Links
- **Installation & Setup**: See [Getting Started](#-getting-started) below
- **Tool Usage Examples**: [COMPLETE_DOCUMENTATION.md §3](./docs/COMPLETE_DOCUMENTATION.md#3-tools-api-reference)
- **Agent Architecture**: [ARCHITECTURE_DIAGRAMS.md §2](./docs/architecture/ARCHITECTURE_DIAGRAMS.md#2-langgraph-agent-flow)
- **Dataset Schemas**: [COMPLETE_DOCUMENTATION.md §4](./docs/COMPLETE_DOCUMENTATION.md#4-dataset-documentation)

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Git**
- **LM Studio** (recommended) or **API Keys** (at least one):
  - [LM Studio](https://lmstudio.ai/) (Free, local, privacy-focused)
  - [Groq](https://console.groq.com/) (Free: 14,400 requests/day)
  - [Google AI Studio](https://aistudio.google.com/apikey) (Free: 60 requests/min)
  - [OpenAI](https://platform.openai.com/api-keys) (Pay-per-use)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Silvestre17/Thesis2025-26_AFGS.git
   cd Thesis2025-26_AFGS
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure LLM provider** (create `.env` file):
   
   **Option A: LM Studio (Recommended - No API Key Required)**
   ```bash
   # 1. Download LM Studio from https://lmstudio.ai/
   # 2. Load model: qwen/qwen3-4b-2507
   # 3. Start local server on port 1234
   # No additional configuration needed!
   ```
   
   **Option B: Cloud Providers (Requires API Keys)**
   ```bash
   # LLM Providers (at least one required)
   GROQ_API_KEY=gsk_...
   GOOGLE_API_KEY=AIza...
   OPENAI_API_KEY=sk-...

   # Optional: LangSmith (for tracing/debugging)
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=lsv2_...
   LANGCHAIN_PROJECT=thesis-lisbon-assistant
   ```

4. **Build vector store** (first-time only):
   ```bash
   python tools/vector_store.py
   ```

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

### Usage Example

```python
from agent.graph import create_assistant

# Create assistant instance (uses LM Studio by default)
assistant = create_assistant()  # or specify provider: create_assistant(provider="groq")

# Ask a question
response = assistant.chat("What's the weather in Lisbon this weekend?")
print(response)

# Plan an itinerary
itinerary = assistant.chat("Create a 2-day cultural itinerary for this weekend")
print(itinerary)
```

## 🧪 Testing

**Test individual components:**

```bash
# Test weather tools
python tools/ipma_api.py

# Test transport tools
python tools/transport_api.py

# Test VisitLisboa search
python tools/visitlisboa_api.py

# Test open data tools
python tools/dados_abertos.py

# Validate vector store
python tools/vector_store.py --test

# Test agent
python agent/graph.py
```

## 📈 Project Statistics

- **Python Modules**: 20+
- **Agent Tools**: 29 (across 4 modules)
- **Specialized Agents**: 5 (Supervisor, Weather, Transport, Researcher, Planner)
- **Data Sources**: 4 real-time APIs + 2 scraped sources
- **Vector DB Documents**: ~1,400 chunks (PDF guide + events + places)
- **Datasets**: 200+ events, 300+ places, 100+ open data GeoJSON
- **Lines of Code**: ~8,000+
- **Documentation**: 150+ pages

## 🤝 Contributing

This is a Master's Thesis project. While direct contributions are not accepted, feedback and suggestions are welcome through GitHub Issues.

## 📧 Contact

**André Filipe Gomes Silvestre**  
Student ID: 20240502  
Email: [TBD]  
Institution: NOVA Information Management School (NOVA IMS)  
Program: Master in Data Science and Advanced Analytics

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NOVA IMS** for providing the academic framework
- **VisitLisboa** for tourism data
- **IPMA** for weather data APIs
- **Metro de Lisboa**, **Carris Metropolitana**, **CP** for transport data
- **Lisboa Aberta** (Open Data Portal) for government datasets
- **LangChain/LangGraph** team for the agent framework
- **ChromaDB** and **HuggingFace** for RAG infrastructure

---

<p align="center">
    <i>Developed as part of the Master's Thesis in Data Science and Advanced Analytics at NOVA IMS (2025-2026)</i>
</p>

<p align="center">
    <a href="https://www.novaims.unl.pt/"><img src="https://img.shields.io/badge/NOVA_IMS-003DA5?style=for-the-badge&logo=university&logoColor=white" alt="NOVA IMS"></a>
</p>
