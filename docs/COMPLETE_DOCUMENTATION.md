# Complete Project Documentation

**Master Thesis - LLM-Powered Urban Exploration**  
**André Filipe Gomes Silvestre** (20240502)  
**NOVA IMS** - 2025/2026

---

## TABLE OF CONTENTS

1. [System Architecture](#1-system-architecture)
2. [Agent Components](#2-agent-components)
3. [Tools API Reference](#3-tools-api-reference)
4. [Dataset Documentation](#4-dataset-documentation)
5. [Processes & Workflows](#5-processes--workflows)
6. [Error Handling](#6-error-handling)
7. [Testing & Validation](#7-testing--validation)

---

# 1. SYSTEM ARCHITECTURE

## 1.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT FRONTEND                       │
│                         (app.py)                                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LANGGRAPH AGENT                           │
│  ┌────────────┐      ┌────────────┐      ┌──────────────┐     │
│  │   State    │◄────►│   Agent    │◄────►│    Tools     │     │
│  │ Management │      │   (ReAct)  │      │    Node      │     │
│  └────────────┘      └────────────┘      └──────────────┘     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    ┌────────┐   ┌──────────┐  ┌────────────┐
    │  LLM   │   │  Tools   │  │  Vector    │
    │Factory │   │  (18)    │  │   Store    │
    └────────┘   └──────────┘  └────────────┘
         │             │              │
         ▼             ▼              ▼
    [Groq/     [APIs: IPMA,    [ChromaDB:
     Google/    Metro, Carris,   PDF, Places,
     OpenAI]    CP, Dados]       Events]
```

## 1.2 Technology Stack

### Core Framework
- **LangGraph**: Agent orchestration (v0.2.0+)
- **LangChain**: LLM abstraction (v0.3.0+)
- **Streamlit**: Web interface (v1.30.0+)

### LLM Providers
- **Groq** (Default): qwen/qwen3-4b-2507
- **Google**: gemini-2.0-flash-exp
- **OpenAI**: gpt-4o-mini
- **Local**: LM Studio, Ollama

### Data Layer
- **ChromaDB**: Vector database (v1.0.0+)
- **HuggingFace**: Embeddings (BAAI/bge-m3)
- **BeautifulSoup4**: Web scraping
- **Pandas**: Data manipulation

### APIs
- **IPMA**: Weather data
- **Metro de Lisboa**: Line status
- **Carris Metropolitana**: Bus alerts/schedules
- **CP**: Train status
- **Dados.gov**: Open government data

---

# 2. AGENT COMPONENTS

## 2.1 Agent Architecture

### File: `agent/graph.py`

**Purpose**: Implements the LangGraph agent using ReAct pattern.

#### Key Components

##### `create_agent_node(llm_with_tools)`
Creates the main reasoning node.

**Process**:
1. Receives current state
2. Adds system prompt if missing
3. Invokes LLM with tools
4. Returns updated state

**Input**:
- `llm_with_tools`: LLM instance with bound tools

**Output**:
- Callable that processes state

##### `should_continue(state: AgentState) -> str`
Conditional routing function.

**Logic**:
```python
if last_message.tool_calls:
    return "tools"  # Continue to tool execution
else:
    return "end"    # End conversation
```

##### `build_agent_graph(provider: str = None)`
Constructs the complete workflow graph.

**Graph Structure**:
```
                  ┌─────────┐
                  │  START  │
                  └────┬────┘
                       │
                       ▼
                  ┌─────────┐
           ┌──────┤  Agent  ├──────┐
           │      └─────────┘      │
           │                       │
      tool_calls              no tool_calls
           │                       │
           ▼                       ▼
      ┌─────────┐              ┌─────┐
      │  Tools  │              │ END │
      └────┬────┘              └─────┘
           │
      (always back to Agent)
           │
           └──────────────┘
```

**Returns**: Compiled graph ready for invocation

#### LisbonAssistant Class

**Purpose**: High-level interface for agent interaction.

**Methods**:

```python
class LisbonAssistant:
    def __init__(self, provider: str = None):
        """Initialize with optional LLM provider."""
        
    def chat(self, message: str) -> str:
        """Send message and get response."""
        
    def reset(self):
        """Reset conversation state."""
        
    def get_history(self) -> List:
        """Get full conversation history."""
```

**Usage Example**:
```python
from agent.graph import create_assistant

# Create assistant
assistant = create_assistant()

# Chat
response = assistant.chat("What's the weather in Lisbon?")
print(response)

# Reset for new conversation
assistant.reset()
```

---

## 2.2 State Management

### File: `agent/state.py`

**Purpose**: Defines typed state schema for agent.

#### State Classes

##### `AgentState` (TypedDict)
Main state container.

**Fields**:
```python
{
    "messages": List[BaseMessage],  # With add_messages reducer
    "user_context": Optional[UserContext],
    "weather_context": Optional[WeatherContext],
    "transport_context": Optional[TransportContext],
    "current_plan": Optional[List[dict]],
    "session_id": Optional[str],
    "last_tool_result": Optional[str]
}
```

##### `UserContext` (TypedDict)
User preferences and location.

```python
{
    "latitude": float,          # User latitude
    "longitude": float,         # User longitude
    "preferences": List[str],   # Interests
    "language": str,            # 'en' or 'pt'
    "available_time": int,      # Hours
    "mobility": str             # 'full', 'limited', 'wheelchair'
}
```

##### `WeatherContext` (TypedDict)
Current weather data.

```python
{
    "temperature_min": float,
    "temperature_max": float,
    "precipitation_prob": float,
    "weather_type": str,
    "has_warnings": bool,
    "warnings": List[str]
}
```

##### `TransportContext` (TypedDict)
Transport status cache.

```python
{
    "metro_status": dict,      # Per-line status
    "carris_alerts": int,      # Count of alerts
    "train_delays": int,       # Count of delays
    "last_updated": str        # ISO timestamp
}
```

#### Helper Functions

```python
def create_initial_state(session_id: str = None) -> AgentState:
    """Create empty initial state."""
    
def update_weather_context(state, temp_min, temp_max, ...) -> AgentState:
    """Update weather data in state."""
    
def update_user_location(state, latitude, longitude) -> AgentState:
    """Update user location."""
```

---

## 2.3 System Prompts

### File: `agent/prompts.py`

**Purpose**: Defines agent personality and constraints.

#### Main System Prompt

**Key Directives**:
1. **MUST use tools** - Never invent information
2. **Tool-first approach** - Call tools before responding
3. **Natural presentation** - Hide tool mechanics from users
4. **Bilingual** - Support English and Portuguese (PT-PT)

**Tool Usage Rules**:
```
Attractions/Museums → search_places_attractions()
Events/Exhibitions → search_cultural_events()
Weather → get_current_weather_summary()
Transport → get_transport_summary()
Services → find_nearby_services()
```

**Response Guidelines**:
- Present tool results naturally
- Use emojis sparingly
- Be concise but complete
- Respond in user's language

#### Specialized Prompts

**`ITINERARY_PLANNING_PROMPT`**:
Step-by-step itinerary creation with:
- Preference understanding
- Weather adaptation
- Logistics optimization

**`WEATHER_ANALYSIS_PROMPT`**:
Weather interpretation with:
- Current conditions
- Active warnings
- Activity recommendations

**`TRANSPORT_ANALYSIS_PROMPT`**:
Transport routing with:
- Real-time status
- Alternative routes
- Travel time estimates

---

## 2.4 LLM Factory

### File: `agent/llm_factory.py`

**Purpose**: Unified LLM instantiation across providers.

#### Design Pattern: Factory Pattern

**Benefits**:
- Provider abstraction
- Centralized configuration
- Easy switching

#### Supported Providers

##### 1. Groq (Default)
**Model**: `qwen/qwen3-4b-2507`  
**Speed**: Extremely fast  
**Free Tier**: 14,400 requests/day  

```python
from agent.llm_factory import LLMFactory

llm = LLMFactory.get_llm(provider="groq")
```

##### 2. Google
**Model**: `gemini-2.0-flash-exp`  
**Features**: Multimodal  
**Free Tier**: 60 requests/minute  

```python
llm = LLMFactory.get_llm(provider="google")
```

##### 3. OpenAI
**Model**: `gpt-4o-mini`  
**Quality**: Highest  
**Pricing**: Pay-per-use  

```python
llm = LLMFactory.get_llm(provider="openai")
```

##### 4. LM Studio (Local)
**Server**: http://localhost:1234/v1  
**Privacy**: Full offline  

```python
llm = LLMFactory.get_llm(provider="lmstudio")
```

##### 5. Ollama (Local)
**Model**: `qwen2.5:7b`  
**Setup**: `ollama pull qwen2.5:7b`  

```python
llm = LLMFactory.get_llm(provider="ollama")
```

#### Methods

```python
@staticmethod
def get_llm(provider: str, temperature: float) -> BaseChatModel:
    """Create LLM instance."""
    
@staticmethod
def get_model_info(llm: BaseChatModel) -> str:
    """Extract model name from instance."""
```

---

# 3. TOOLS API REFERENCE

## 3.1 Weather Tools (IPMA)

### File: `tools/ipma_api.py`

#### `get_weather_warnings(area: str = "LSB") -> str`

**Purpose**: Fetch active weather warnings for Lisbon.

**Parameters**:
- `area` (str): Area code (default: "LSB" for Lisbon)

**Returns**: Formatted warning list or "No warnings" message

**API**: `https://api.ipma.pt/open-data/forecast/warnings/warnings_www.json`

**Warning Levels**:
- 🟢 Green: No warning (filtered out)
- 🟡 Yellow: Be aware
- 🟠 Orange: Be prepared
- 🔴 Red: Take action

**Example Output**:
```
⚠️ Active Weather Warnings for Lisbon:

🟡 WIND (Be aware)
   ⏰ Dec 30, 14:00 to Dec 31, 06:00
   📝 Strong winds expected in coastal areas

💡 Check IPMA.pt for detailed information.
```

**Error Handling**:
- Timeout: 10s with no retry (fast failure)
- Invalid JSON: Returns error message
- Empty data: Returns "No warnings"

---

#### `get_weather_forecast(days: int = 3) -> str`

**Purpose**: Get daily forecast for Lisbon (1-5 days).

**Parameters**:
- `days` (int): Number of days (1-5, default: 3)

**Returns**: Formatted forecast with temperatures, precipitation, wind

**API**: `https://api.ipma.pt/open-data/forecast/meteorology/cities/daily/{global_id}.json`

**Lisbon ID**: `1110600` (from `Config.LISBON_GLOBAL_ID`)

**Weather Type Mapping**: 27 IPMA codes to descriptions

**Example Output**:
```
🌤️ Weather Forecast for Lisbon
========================================
📅 Updated: 2025-12-30T06:00:00

☀️ Monday, Dec 30
   🌡️ 12°C to 18°C
   🌤️ Partly cloudy
   💧 Rain: Unlikely (15%)
   💨 Wind: Northwest

🌧️ Tuesday, Dec 31
   🌡️ 14°C to 16°C
   🌤️ Rain
   💧 Rain: Likely (75%)
   💨 Wind: Southwest
```

---

#### `get_current_weather_summary() -> str`

**Purpose**: Quick summary of today's weather + warnings.

**Parameters**: None

**Returns**: Combined today's forecast and active warnings

**Use Case**: First call for weather queries

**Example Output**:
```
🌤️ Lisbon Weather Summary
========================================

📅 Today (2025-12-30):
   🌡️ Temperature: 12°C to 18°C
   🌤️ Conditions: Partly cloudy
   💧 Rain probability: 15%

✅ No active weather warnings.
```

---

## 3.2 Transport Tools

### File: `tools/transport_api.py`

#### Metro de Lisboa

##### `get_metro_status() -> str`

**Purpose**: Real-time status of all 4 metro lines.

**Returns**: Operational status for each line

**API**: `https://app.metrolisboa.pt/status/getLinhas.php`

**Lines**:
- 🟡 Yellow (Rato ↔ Odivelas)
- 🔵 Blue (Santa Apolónia ↔ Reboleira)
- 🟢 Green (Telheiras ↔ Cais do Sodré)
- 🔴 Red (S. Sebastião ↔ Aeroporto)

**Example Output**:
```
🚇 Metro de Lisboa Status
========================================

🟡 Yellow Line (Rato ↔ Odivelas)
   ✅ Normal service

🔵 Blue Line (Santa Apolónia ↔ Reboleira)
   ⚠️ Delays due to technical issues

🟢 Green Line (Telheiras ↔ Cais do Sodré)
   ✅ Normal service

🔴 Red Line (S. Sebastião ↔ Aeroporto)
   ✅ Normal service

⚠️ Some lines have service disruptions.
```

---

#### Carris Metropolitana (Buses)

##### `get_carris_alerts() -> str`

**Purpose**: Active service alerts for bus network.

**Returns**: Alert descriptions with affected routes

**API**: `https://api.carrismetropolitana.pt/v2/alerts`

**Alert Effects**:
- 🚫 NO_SERVICE: Complete suspension
- ⚠️ REDUCED_SERVICE: Limited frequency
- 🕐 SIGNIFICANT_DELAYS: Major delays
- ↩️ DETOUR: Route changes
- 📍 STOP_MOVED: Stop relocation

**Example Output**:
```
🚌 Carris Metropolitana Alerts (3 active)
==================================================

1. ↩️ Route Deviation - Line 728
   📍 Routes: 728, 729, 730
   🔸 Cause: Construction Work
   🔸 Effect: Detour
   📝 Temporary route change due to road works

2. ⚠️ Service Reduction
   📍 Routes: 705, 710
   🔸 Cause: Weather
   🔸 Effect: Reduced Service
```

##### `get_carris_stop_info(stop_id: str) -> str`

**Purpose**: Bus stop details + real-time arrivals.

**Parameters**:
- `stop_id` (str): Stop ID (e.g., "060001")

**Returns**: Stop info with next arrivals

**Example Output**:
```
🚏 Bus Stop Information
========================================

📍 Praça do Comércio
   📌 Lisbon
   🗺️ (38.7076, -9.1365)
   🚌 Lines: 728, 711, 714, 732, 735

⏱️ Upcoming Arrivals:
   1. Line 728 → Portela (Aeroporto)
      ⏰ 3 min
   2. Line 711 → Amadora
      ⏰ 12 min
```

---

## 3.3 Open Data Tools

### File: `tools/dados_abertos.py`

#### `find_nearby_services(service_type, user_lat, user_lon, max_results) -> str`

**Purpose**: Find public services with proximity filtering.

**Parameters**:
- `service_type` (str): Service type (farmácias, hospitais, escolas, wifi, metro, jardins)
- `user_lat` (float, optional): User latitude
- `user_lon` (float, optional): User longitude
- `max_results` (int): Maximum results (default: 5)

**Process**:
1. Search metadata for matching dataset
2. Fetch GeoJSON from stable URL (with retry)
3. Extract coordinates from features
4. Calculate Haversine distances
5. Sort by proximity
6. Return top N results

**Example Output**:
```
📍 Found 5 results from 'Farmácias de Lisboa':

1. Farmácia Central
   📍 Rua Augusta 125, Lisboa
   📏 0.32 km away
   🗺️ (38.7125, -9.1386)

2. Farmácia Barreiros
   📍 Praça da Figueira 7, Lisboa
   📏 0.58 km away
   🗺️ (38.7141, -9.1389)
```

**Error Handling**:
- Retry logic: 3 attempts with exponential backoff (2s, 4s, 8s)
- Timeout: 15s per request
- GeoJSON validation before processing

---

## 3.4 VisitLisboa Tools

### File: `tools/visitlisboa_api.py`

#### `search_cultural_events(query, category, date_filter, max_results) -> str`

**Purpose**: Search events with CRITICAL date filtering.

**Parameters**:
- `query` (str, optional): Natural language search
- `category` (str, optional): Event category filter
- `date_filter` (str, optional): Date range (default: "upcoming")
- `max_results` (int): Max results (default: 10)

**Date Parsing**:
Supports natural language:
- `"today"`, `"tomorrow"`
- `"this week"`, `"next week"`
- `"this weekend"`, `"this month"`
- `"January"`, `"February"`, etc.
- `"2025-01-15"` (ISO format)
- `"upcoming"` (default: next 30 days)

**Search Strategy**:
1. Filter by date FIRST (most important)
2. Filter by category (if specified)
3. Keyword match in title/description

**Example Output**:
```
🎭 Found 12 Cultural Events in Lisbon:
📅 Date range: this weekend (2025-12-28 to 2025-12-30)
📆 Today is: Saturday, 28 December 2025

1. 📅 Christmas Concert at Belém
   🗓️ When: 28 Dec, 2025 at 20:00
   📂 Category: Music
   Classical orchestra performs holiday favorites
   📍 CCB - Centro Cultural de Belém
   🔗 https://www.visitlisboa.com/en/events/...

2. 📅 Modern Art Exhibition Opening
   🗓️ When: 29 Dec, 2025 to 15 Jan, 2026
   📂 Category: Exhibitions
   Contemporary Portuguese artists
   📍 MAAT - Museum of Art, Architecture and Technology
   🔗 https://www.visitlisboa.com/en/events/...
```

**Real Examples from Dataset**:

*Example 1: Misty Fest*
```json
{
  "url": "https://www.visitlisboa.com/en/events/misty-fest-2",
  "category": "Main Events",
  "full_description": "From songwriters' soundscapes to world music and jazz, Misty Fest is truly a different festival...",
  "dates": [],  // No dates in this example
  "information_links": {
    "www.misty-fest.com": "http://www.misty-fest.com"
  }
}
```

*Example 2: Diverlândia*
```json
{
  "url": "https://www.visitlisboa.com/en/events/diverlandia",
  "category": "Others",
  "dates": [
    {
      "type": "range",
      "start": {
        "datetime_iso": "2025-12-12",
        "display_text": "12 Dec, 2025"
      },
      "end": {
        "datetime_iso": "2026-01-04",
        "display_text": "04 Jan, 2026"
      }
    }
  ],
  "venue_name": "FIL - Lisbon Exhibition and Congress Centre",
  "location": "Edifício FIL, Rua do Bojador, Parque das Nações"
}
```

---

#### `search_places_attractions(query, category, max_results) -> str`

**Purpose**: Semantic search for places using AI embeddings.

**Parameters**:
- `query` (str, optional): Natural language description
- `category` (str, optional): Place category
- `max_results` (int): Max results (default: 10)

**Search Method**:
1. **Primary**: Semantic search via vector store (ChromaDB)
2. **Fallback**: Keyword search in JSON if vector store unavailable

**Categories**:
Museums & Monuments, Restaurants, Hotels, View Points, Beaches, Shopping, Nightlife, Parks & Gardens, Tours

**Example Output**:
```
🏛️ Found 5 Places/Attractions in Lisbon:

1. 🏛️ **National Museum of Archaeology**
   Category: Museums & Monuments
   The National Museum of Archaeology has important collections...
   📍 Praça do Império, 1400, Lisboa
   🔗 https://www.visitlisboa.com/en/places/...
```

**Real Example from Dataset**:
```json
{
  "url": "https://www.visitlisboa.com/en/places/national-museum-of-archaeology",
  "title": "National Museum of Archaeology",
  "category": "Museums & Monuments",
  "short_description": "The National Museum of Archaeology has important collections...",
  "contact_info": {
    "phone": "351213620000",
    "email": "geral@mnarqueologia.dgpc.pt",
    "website": "http://www.museunacionalarqueologia.gov.pt/"
  },
  "schedule": {
    "today": "Today: Closed",
    "Monday": "Closed",
    "Tuesday": "Closed",
    ...
  },
  "location": "Praça do Império, 1400, Lisboa",
  "tripadvisor": {
    "rating": "3.5",
    "reviews_count": "254",
    "url": "https://www.tripadvisor.com/..."
  }
}
```

---

## 3.5 Vector Store

### File: `tools/vector_store.py`

**Purpose**: RAG (Retrieval-Augmented Generation) knowledge base.

#### ChromaDB Collections

##### 1. `lisbon_pdf`
**Source**: TurismodeLisboa_OfficialGuide.pdf  
**Chunks**: ~900 chunks (1000 chars, 200 overlap)  
**Update**: Static (index once)  

##### 2. `lisbon_places`
**Source**: places.json  
**Documents**: ~300 places  
**Update**: Weekly sync  

##### 3. `lisbon_events`
**Source**: events.json  
**Documents**: ~200 events  
**Update**: Daily sync  

#### Incremental Sync

**Key Innovation**: Only processes changed documents.

**Algorithm**:
1. Load current JSON data
2. Compute content hashes (SHA-256)
3. Compare with existing DB hashes
4. Identify: new, modified, deleted
5. Delete: modified + deleted
6. Add: new + modified

**Example Sync Output**:
```
==========================================================
🔄 Vector Store Incremental Sync
==========================================================

📁 VisitLisboa_Events Collection (lisbon_events)
   📂 Loaded 203 items from JSON
   📊 Existing in DB: 198 items
   ➕ New: 12
   🔄 Modified: 3
   ➖ Deleted: 7
   🗑️ Deleted 10 documents from DB
   ✓ Added/Updated 15 documents

📊 Sync Summary
==========================================================
   events: ✓ Synced (+12 ~3 -7 = 203 docs)
   places: ✓ No changes (301 docs)
   pdf: ✓ Skipped (912 docs)
```

#### Search Function

```python
def search(query: str, k: int = 5, collections: List[str] = None) -> List[Document]:
    """Semantic search across collections."""
```

**Process**:
1. Generate query embedding (BAAI/bge-m3)
2. Search specified collections
3. Compute cosine similarity
4. Return top-k documents sorted by relevance

---

# 4. DATASET DOCUMENTATION

## 4.1 VisitLisboa Events

**File**: `data_collection/webscraping/events.json`

### Schema

```json
{
  "url": "string",                    // Event page URL
  "category": "string",               // Event category
  "image_urls": ["string"],           // Image URLs
  "video_urls": ["string"],           // Video URLs (often empty)
  "full_description": "string",       // Full HTML-parsed description
  "dates": [                          // Event dates array
    {
      "type": "single|range",         // Date type
      "date": {                       // For single dates
        "datetime_iso": "YYYY-MM-DD", // ISO format
        "display_text": "string",     // Human-readable
        "time": "HH:MM"               // Optional time
      },
      "start": {...},                 // For ranges
      "end": {...}
    }
  ],
  "information_links": {              // External links
    "display_text": "url"
  },
  "buy_tickets_url": "string|null",   // Ticket purchase link
  "venue_name": "string",             // Venue (if available)
  "location": "string"                // Address (if available)
}
```

### Real Examples

**Example 1: Event Without Dates**
```json
{
  "url": "https://www.visitlisboa.com/en/events/misty-fest-2",
  "category": "Main Events",
  "image_urls": ["https://www.visitlisboa.com/rails/active_storage/..."],
  "video_urls": [],
  "full_description": "From songwriters' soundscapes to world music and jazz...",
  "dates": [],
  "information_links": {
    "www.misty-fest.com": "http://www.misty-fest.com",
    "https://www.facebook.com/MistyFest?fref=ts": "https://www.facebook.com/..."
  },
  "buy_tickets_url": null
}
```

**Example 2: Event With Date Range**
```json
{
  "url": "https://www.visitlisboa.com/en/events/diverlandia",
  "category": "Others",
  "full_description": "Fun for all ages, with family or friends...",
  "dates": [
    {
      "type": "range",
      "start": {
        "datetime_iso": "2025-12-12",
        "display_text": "12 Dec, 2025",
        "time": null
      },
      "end": {
        "datetime_iso": "2026-01-04",
        "display_text": "04 Jan, 2026",
        "time": null
      }
    }
  ],
  "venue_name": "FIL - Lisbon Exhibition and Congress Centre",
  "location": "Edifício FIL, Rua do Bojador, Parque das Nações, 1998-010, Lisboa"
}
```

### Categories
- Main Events
- Exhibitions
- Music
- Theater
- Dance
- Cinema
- Sports
- Fairs
- Festivals
- Gastronomy
- Others

### Statistics
- **Total Events**: ~200
- **With Dates**: ~80%
- **Average Description Length**: 300-500 characters
- **Update Frequency**: Daily (GitHub Actions)

---

## 4.2 VisitLisboa Places

**File**: `data_collection/webscraping/places.json`

### Schema

```json
{
  "url": "string",
  "title": "string",
  "category": "string",
  "short_description": "string",
  "image_urls": ["string"],
  "video_urls": ["string"],
  "full_description": "string",
  "features": ["string"],              // Amenities/features
  "contact_info": {
    "phone": "string",
    "email": "string",
    "website": "string"
  },
  "social_media": {
    "platform": "url"
  },
  "schedule": {
    "today": "string",
    "Monday": "string",
    ...
  },
  "location": "string",                // Full address
  "tripadvisor": {
    "rating": "string",
    "reviews_count": "string",
    "url": "string"
  }
}
```

### Real Example

```json
{
  "url": "https://www.visitlisboa.com/en/places/national-museum-of-archaeology",
  "title": "National Museum of Archaeology",
  "category": "Museums & Monuments",
  "short_description": "The National Museum of Archaeology has important collections of the Portuguese archeology, dating from prehistory to medieval times.",
  "image_urls": [
    "https://www.visitlisboa.com/rails/active_storage/.../museu-nacional-de-arqueologia-0.jpg",
    "https://www.visitlisboa.com/rails/active_storage/.../museu-nacional-de-arqueologia-1.jpg"
  ],
  "video_urls": [],
  "full_description": "The National Museum of Archaeology has a vast collection and was designed...",
  "features": [],
  "contact_info": {
    "phone": "351213620000",
    "email": "geral@mnarqueologia.dgpc.pt",
    "website": "http://www.museunacionalarqueologia.gov.pt/"
  },
  "social_media": {
    "/assets/icons/social/twitter": "https://twitter.com/MNArqueologia",
    "/assets/icons/social/youtube": "https://www.youtube.com/channel/..."
  },
  "schedule": {
    "today": "Today: Closed",
    "Sunday": "Closed",
    "Monday": "Closed",
    "Tuesday": "Closed",
    "Wednesday": "Closed",
    "Thursday": "Closed",
    "Friday": "Closed",
    "Saturday": "Closed"
  },
  "location": "Praça do Império, 1400, Lisboa",
  "tripadvisor": {
    "rating": "3.5",
    "reviews_count": "254",
    "url": "https://www.tripadvisor.com/UserReview-g189158-d3238655-..."
  }
}
```

### Categories
- Museums & Monuments
- Restaurants
- Hotels
- View Points
- Beaches
- Shopping
- Nightlife
- Parks & Gardens
- Tours

### Statistics
- **Total Places**: ~300
- **With Contact Info**: ~85%
- **With Schedules**: ~70%
- **With TripAdvisor**: ~60%
- **Update Frequency**: Weekly (GitHub Actions)

---

## 4.3 Lisboa Aberta (Open Data)

**File**: `data_collection/webscraping/lisbon_datasets_clean.json`

### Schema

```json
{
  "title": "string",
  "url_portal": "string",              // Portal page URL
  "stable_url": "string",              // Direct GeoJSON URL
  "description": "string",
  "file_formats": "string",            // Usually "geojson"
  "last_updated": "ISO_8601_datetime"
}
```

### Real Example

```json
{
  "title": "Lisboa. Pontos de encontro - Emergência.",
  "url_portal": "https://dados.gov.pt/pt/datasets/plano-municipal-de-emergencia-em-protecao-civil-pmepc/",
  "stable_url": "https://dados.gov.pt/pt/datasets/r/1df28fd3-9c10-4cde-9954-5397e275e333",
  "description": "Serviço de mapa com indicação da localização de proteção civil e na Cidade de Lisboa. (PMEPC_PE_PT).",
  "file_formats": "geojson",
  "last_updated": "2025-10-30T00:00:00"
}
```

### Common Datasets
- **Farmácias**: Pharmacy locations
- **Hospitais**: Hospital locations
- **Escolas**: School locations
- **Jardins/Parques**: Parks and gardens
- **WiFi**: Public WiFi hotspots
- **Metro**: Metro station locations
- **Fontanários**: Public water fountains
- **Estacionamento**: Parking locations

### Statistics
- **Total Datasets**: ~100
- **All GeoJSON**: 100%
- **Validation**: All tested and accessible
- **Update Frequency**: Static metadata, dynamic fetch

---

# 5. PROCESSES & WORKFLOWS

## 5.1 Web Scraping

### VisitLisboa Events Scraper

**File**: `data_collection/webscraping/visitlisbon_events.py`

**Process**:
1. **Determine Total Pages**: Parse pagination (`?page=N`)
2. **Collect Event URLs**: Scrape all listing pages
3. **Visit Each Event**: Extract detailed information
4. **Parse Dates**: Complex date extraction logic
5. **Incremental Update**: Compare with existing JSON
6. **Save**: Write to `events.json`

**Anti-Bot Measures**:
- Random User-Agent rotation
- Random delays (2-4s between requests)
- Respectful scraping (not concurrent)

**Date Parsing Logic**:
```python
# Handles various formats:
# - "28 Dec, 2025"
# - "28 Dec, 2025 to 04 Jan, 2026"
# - "Saturdays and Sundays"
# - "28 Dec, 2025 at 20:00"
```

**Incremental Update**:
```python
# Only scrapes new events or updates changed ones
# Preserves existing data for unchanged events
```

---

### VisitLisboa Places Scraper

**File**: `data_collection/webscraping/visitlisbon_places.py`

**Process**:
Similar to events, but extracts:
- Contact information
- Social media links
- Operating schedules
- TripAdvisor data

**Schedule Parsing**:
```python
# Extracts daily hours:
# "Monday": "10:00 - 18:00"
# "Tuesday": "Closed"
```

---

### Dados Abertos Scraper

**File**: `data_collection/webscraping/dadosabertos.gov_lisboa.py`

**Process**:
1. **Search Page**: Filter by Lisbon (geozone=pt:concelho:1106)
2. **Parse Results**: Extract dataset cards
3. **Visit Dataset Page**: Get detailed info
4. **Extract Stable URL**: Direct GeoJSON link
5. **Metadata Only**: Don't download GeoJSON (fetched on-demand)

**Filtering**:
- Skip "Resultados" (search results)
- Skip "Desafio" (challenges/competitions)

---

## 5.2 Vector Store Synchronization

**File**: `tools/vector_store.py`

### Sync Algorithm

```
FOR each collection (pdf, places, events):
    1. Load current JSON data
    2. Compute content hashes for all items
    3. Get existing document IDs and hashes from ChromaDB
    4. Compare:
        new_ids = current_ids - existing_ids
        deleted_ids = existing_ids - current_ids
        modified_ids = {id: hash != existing_hash}
    5. DELETE: modified_ids ∪ deleted_ids
    6. ADD: new_ids ∪ modified_ids
    7. Report statistics
```

### Content Hashing

```python
def compute_content_hash(content: str) -> str:
    """SHA-256 hash of content (first 16 chars)."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
```

**Why?** Detects changes in event descriptions, schedule updates, etc.

### Document ID Generation

```python
def generate_doc_id(url: str, source: str) -> str:
    """Stable ID from URL."""
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:12]
    return f"{source}_{url_hash}"
```

**Why?** Ensures same event/place always gets same ID across syncs.

---

## 5.3 GitHub Actions Automation

### Workflow 1: Vector Store Sync

**File**: `.github/workflows/sync_vector_store.yml`

**Schedule**: Daily at 3 AM UTC

**Steps**:
1. Checkout repository
2. Set up Python environment
3. Install dependencies (CPU-only PyTorch)
4. Run `python tools/vector_store.py`
5. Commit changes if any
6. Push to repository

**Caching**:
- pip packages (~500MB)
- Embedding model (~600MB)

---

### Workflow 2: Web Scraping

**File**: `.github/workflows/scrape_visitlisboa.yml`

**Schedule**: Daily at 2 AM UTC

**Steps**:
1. Checkout repository
2. Set up Python
3. Install scraping dependencies
4. Run event scraper (Mondays, Wednesdays, Fridays)
5. Run place scraper (Sundays only)
6. Commit updated JSON files
7. Trigger vector store sync

---

# 6. ERROR HANDLING

## 6.1 Network Errors

### Retry Strategy

**Pattern**: Exponential Backoff

```python
for attempt in range(MAX_RETRIES):
    try:
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        wait_time = BACKOFF_FACTOR ** attempt  # 2^0, 2^1, 2^2 = 1s, 2s, 4s
        logger.warning(f"Timeout. Retrying in {wait_time}s...")
        time.sleep(wait_time)
```

**Timeouts**:
- IPMA: 10s (fast API)
- Transport: 15s (larger payloads)
- GeoJSON: 15s (variable size)

---

## 6.2 API Errors

### HTTP Status Codes

```python
try:
    response.raise_for_status()  # Raises HTTPError for 4xx/5xx
except requests.exceptions.HTTPError as e:
    if response.status_code == 404:
        return "Resource not found"
    elif response.status_code == 429:
        return "Rate limit exceeded. Try again later."
    elif response.status_code >= 500:
        return "Server error. Service temporarily unavailable."
```

---

## 6.3 Data Validation

### GeoJSON Validation

```python
def is_valid_geojson(data: Any) -> bool:
    """Validates GeoJSON structure."""
    if not isinstance(data, dict):
        return False
    if "type" not in data:
        return False
    valid_types = [
        "FeatureCollection", "Feature", "Point", "LineString",
        "Polygon", "MultiPoint", "MultiLineString", "MultiPolygon"
    ]
    return data["type"] in valid_types
```

---

### Date Parsing Fallback

```python
# Try multiple formats
try:
    dt = datetime.strptime(date_str, '%Y-%m-%d')
except ValueError:
    try:
        dt = datetime.strptime(date_str, '%d/%m/%Y')
    except ValueError:
        return None  # Unable to parse
```

---

## 6.4 Vector Store Errors

### ChromaDB Connection

```python
try:
    vectorstore = Chroma(...)
except Exception as e:
    logger.warning(f"Vector store unavailable: {e}")
    # Fall back to JSON keyword search
    return _fallback_search(query, data, max_results)
```

---

# 7. TESTING & VALIDATION

## 7.1 Unit Testing

### Test Blocks

Each module includes standalone test:

```python
if __name__ == "__main__":
    # Test code here
```

**Run Tests**:
```bash
python tools/ipma_api.py
python tools/transport_api.py
python tools/dados_abertos.py
python tools/visitlisboa_api.py
python tools/vector_store.py --test
python agent/graph.py
```

---

## 7.2 Vector Store Validation

**Command**: `python tools/vector_store.py --test`

**Checks**:
1. **Metadata Completeness**: All required fields present
2. **Field Validation**: No "N/A" or empty critical fields
3. **Search Quality**: Queries return relevant results
4. **Collection Integrity**: Document counts match expectations

**Example Output**:
```
📋 Metadata Validation by Collection:

lisbon_events (203 docs)
  1. Title: Misty Fest 2
     URL: https://www.visitlisboa.com/en/events/misty-fest-2
     Category: Main Events
  ✓ All metadata fields valid

lisbon_places (301 docs)
  1. Title: National Museum of Archaeology
     URL: https://www.visitlisboa.com/en/places/...
     Category: Museums & Monuments
  ⚠️ Empty/Invalid fields: schedule (some days)

🔍 Search Quality Test:
  📝 Query: "museums in Belém"
  Expected: Should return places/PDF about Belém museums
  1. [✓] Jerónimos Monastery (VisitLisboa_Places)
  2. [✓] Belém Tower (VisitLisboa_Places)
  3. [✓] MAAT Museum (VisitLisboa_Places)
```

---

## 7.3 Integration Testing

### Agent Test Flow

```bash
python agent/graph.py
```

**Tests**:
1. Agent initialization
2. Weather query
3. Metro status query
4. Reset functionality

**Expected Behavior**:
- Tools called automatically
- Results formatted naturally
- No error messages (unless APIs down)

---

## 7.4 Data Quality Checks

### GeoJSON URL Validation

**Notebook**: `data_collection/webscraping/VisitLisbon_DadosAbertosLx.ipynb`

**Process**:
```python
for dataset in datasets:
    geojson = fetch_geojson(dataset['stable_url'])
    if geojson and is_valid_geojson(geojson):
        success_count += 1
    else:
        failed_urls.append(dataset['stable_url'])
```

**Results**:
- ✅ Success: 98/100 datasets
- ❌ Failed: 2/100 (timeout or invalid)

---

*This completes the comprehensive documentation. For specific implementation questions, refer to the source code with these docs as a guide.*

---

**Last Updated**: December 30, 2025  
**Author**: André Filipe Gomes Silvestre (20240502)  
**Institution**: NOVA IMS
