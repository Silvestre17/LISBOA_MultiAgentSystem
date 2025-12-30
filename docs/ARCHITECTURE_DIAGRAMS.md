# System Architecture Diagrams

**Project**: LLM-Powered Urban Exploration  
**Author**: André Filipe Gomes Silvestre

---

## 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                             │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    Streamlit Web App                          │ │
│  │  - Chat Interface                                             │ │
│  │  - Multi-language Support (EN/PT)                             │ │
│  │  - Provider Selection                                         │ │
│  │  - Session Management                                         │ │
│  └───────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                │ User Messages
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        LANGGRAPH AGENT                              │
│                                                                     │
│  ┌──────────────┐      ┌──────────────┐      ┌─────────────────┐  │
│  │              │      │              │      │                 │  │
│  │    STATE     │◄────►│    AGENT     │◄────►│     TOOLS       │  │
│  │  MANAGEMENT  │      │   (ReAct)    │      │      NODE       │  │
│  │              │      │              │      │                 │  │
│  └──────────────┘      └──────────────┘      └─────────────────┘  │
│        │                     │                       │             │
│        │                     │                       │             │
│        │                     ▼                       │             │
│        │            ┌──────────────┐                 │             │
│        │            │  SYSTEM      │                 │             │
│        └───────────►│  PROMPT      │                 │             │
│                     └──────────────┘                 │             │
└──────────────────────────────────────────────────────┼─────────────┘
                                                       │
                                                       │
            ┌──────────────────────────────────────────┼──────────┐
            │                                          │          │
            ▼                                          ▼          ▼
     ┌──────────────┐                          ┌────────────────────┐
     │  LLM FACTORY │                          │   TOOL MODULES     │
     └──────────────┘                          └────────────────────┘
            │                                          │
            │ Provider Selection                       │ Data Fetching
            ▼                                          ▼
     ┌──────────────┐                          ┌────────────────────┐
     │   LLM APIs   │                          │   DATA SOURCES     │
     │              │                          │                    │
     │ - Groq       │                          │ - IPMA (Weather)   │
     │ - Google     │                          │ - Metro Lisboa     │
     │ - OpenAI     │                          │ - Carris Metro     │
     │ - Local      │                          │ - CP (Trains)      │
     └──────────────┘                          │ - Dados Abertos    │
                                               │ - Vector Store     │
                                               └────────────────────┘
```

---

## 2. LangGraph Agent Flow

```
START
  │
  ▼
┌─────────────────┐
│  User Message   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Agent Node                        │
│                                     │
│  1. Load conversation state         │
│  2. Add system prompt (if missing)  │
│  3. Invoke LLM with tools           │
│  4. Return response                 │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Conditional Edge                  │
│                                     │
│   Has tool_calls?                   │
└────────┬─────────────┬──────────────┘
         │             │
    YES  │             │  NO
         │             │
         ▼             ▼
┌─────────────────┐   ┌─────────────┐
│   Tools Node    │   │     END     │
│                 │   │             │
│  1. Execute     │   │  Return to  │
│     tools       │   │    User     │
│  2. Collect     │   └─────────────┘
│     results     │
└────────┬────────┘
         │
         │ (always loops back)
         │
         └──────────► Agent Node
```

---

## 3. Tool Execution Workflow

```
User Query: "What's the weather in Lisbon today?"
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Agent Reasoning (LLM)                          │
│                                                 │
│  "I need to call get_current_weather_summary()" │
└────────────────┬────────────────────────────────┘
                 │
                 ▼ tool_call
┌─────────────────────────────────────────────────┐
│  Tool Execution                                 │
│                                                 │
│  Function: get_current_weather_summary()        │
│  Parameters: {}                                 │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  HTTP Request to IPMA API                       │
│                                                 │
│  GET https://api.ipma.pt/.../1110600.json       │
│  GET https://api.ipma.pt/.../warnings_www.json  │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Data Processing                                │
│                                                 │
│  - Parse JSON                                   │
│  - Format temperatures                          │
│  - Check warnings                               │
│  - Build human-readable response                │
└────────────────┬────────────────────────────────┘
                 │
                 ▼ tool_result
┌─────────────────────────────────────────────────┐
│  Agent Formatting (LLM)                         │
│                                                 │
│  "Today in Lisbon: 12-18°C, partly cloudy..."   │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
              User Response
```

---

## 4. Vector Store Synchronization

```
GitHub Actions Trigger (Daily 2 AM UTC)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Web Scraping Jobs                              │
│                                                 │
│  ┌─────────────────┐    ┌──────────────────┐   │
│  │ VisitLisboa     │    │ VisitLisboa      │   │
│  │ Events Scraper  │    │ Places Scraper   │   │
│  └────────┬────────┘    └────────┬─────────┘   │
│           │                      │             │
│           └──────────┬───────────┘             │
└──────────────────────┼─────────────────────────┘
                       │
                       ▼ Updated JSON files
┌─────────────────────────────────────────────────┐
│  Vector Store Sync (3 AM UTC)                   │
│                                                 │
│  FOR each collection:                           │
│                                                 │
│    ┌──────────────────────────────┐             │
│    │ 1. Load JSON Data            │             │
│    └──────────┬───────────────────┘             │
│               ▼                                 │
│    ┌──────────────────────────────┐             │
│    │ 2. Compute Content Hashes    │             │
│    └──────────┬───────────────────┘             │
│               ▼                                 │
│    ┌──────────────────────────────┐             │
│    │ 3. Get Existing DB Hashes    │             │
│    └──────────┬───────────────────┘             │
│               ▼                                 │
│    ┌──────────────────────────────┐             │
│    │ 4. Identify Changes          │             │
│    │    - New IDs                 │             │
│    │    - Modified IDs (hash ≠)   │             │
│    │    - Deleted IDs             │             │
│    └──────────┬───────────────────┘             │
│               ▼                                 │
│    ┌──────────────────────────────┐             │
│    │ 5. Delete Old Documents      │             │
│    │    (modified + deleted)      │             │
│    └──────────┬───────────────────┘             │
│               ▼                                 │
│    ┌──────────────────────────────┐             │
│    │ 6. Add New Documents         │             │
│    │    (new + modified)          │             │
│    └──────────┬───────────────────┘             │
│               ▼                                 │
│    ┌──────────────────────────────┐             │
│    │ 7. Commit Changes            │             │
│    └──────────────────────────────┘             │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 5. Data Sources Integration

```
┌──────────────────────────────────────────────────────────┐
│                    REAL-TIME APIs                        │
└──────────────────────────────────────────────────────────┘
    │           │            │            │
    │           │            │            │
    ▼           ▼            ▼            ▼
┌────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐
│  IPMA  │ │  Metro  │ │  Carris  │ │   CP    │
│Weather │ │ Status  │ │   Bus    │ │ Trains  │
└────────┘ └─────────┘ └──────────┘ └─────────┘
    │           │            │            │
    └───────────┴────────────┴────────────┘
                      │
                      ▼
              Direct API Calls
               (No Caching)
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│                  AGENT TOOLS LAYER                       │
│  - get_weather_forecast()                                │
│  - get_metro_status()                                    │
│  - get_carris_alerts()                                   │
│  - get_train_status()                                    │
└──────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────┐
│                 ON-DEMAND FETCH                          │
└──────────────────────────────────────────────────────────┘
                      │
                      ▼
                ┌──────────┐
                │  Dados   │
                │  Abertos │
                │ (GeoJSON)│
                └──────────┘
                      │
                      ▼
         Metadata in JSON → Fetch on request
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│                  AGENT TOOLS LAYER                       │
│  - find_nearby_services()                                │
│  - list_available_datasets()                             │
│  - get_dataset_details()                                 │
└──────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────┐
│             STATIC + SEMANTIC SEARCH                     │
└──────────────────────────────────────────────────────────┘
    │                    │                  │
    ▼                    ▼                  ▼
┌─────────┐      ┌──────────┐      ┌──────────┐
│ Events  │      │  Places  │      │   PDF    │
│  JSON   │      │   JSON   │      │  Guide   │
└─────────┘      └──────────┘      └──────────┘
    │                    │                  │
    └────────────────────┴──────────────────┘
                         │
                         ▼
               ┌─────────────────┐
               │   ChromaDB      │
               │  Vector Store   │
               │                 │
               │ - Embeddings    │
               │ - Collections   │
               │ - Semantic      │
               │   Search        │
               └─────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│                  AGENT TOOLS LAYER                       │
│  - search_cultural_events()                              │
│  - search_places_attractions()                           │
│  - search_lisbon_knowledge()                             │
└──────────────────────────────────────────────────────────┘
```

---

## 6. State Flow Through Agent

```
┌─────────────────────────────────────────────────────┐
│  Initial State (Session Start)                      │
│                                                     │
│  {                                                  │
│    messages: [],                                    │
│    user_context: null,                              │
│    weather_context: null,                           │
│    transport_context: null,                         │
│    current_plan: null,                              │
│    session_id: "a3f9c2b1"                           │
│  }                                                  │
└────────────────┬────────────────────────────────────┘
                 │
                 │ User: "What's the weather?"
                 ▼
┌─────────────────────────────────────────────────────┐
│  State After User Message                           │
│                                                     │
│  {                                                  │
│    messages: [                                      │
│      HumanMessage("What's the weather?")            │
│    ],                                               │
│    ...                                              │
│  }                                                  │
└────────────────┬────────────────────────────────────┘
                 │
                 │ Agent adds system prompt
                 ▼
┌─────────────────────────────────────────────────────┐
│  State With System Prompt                           │
│                                                     │
│  {                                                  │
│    messages: [                                      │
│      SystemMessage("You are Lisbon Assistant..."),  │
│      HumanMessage("What's the weather?")            │
│    ],                                               │
│    ...                                              │
│  }                                                  │
└────────────────┬────────────────────────────────────┘
                 │
                 │ LLM generates tool call
                 ▼
┌─────────────────────────────────────────────────────┐
│  State After LLM Call                               │
│                                                     │
│  {                                                  │
│    messages: [                                      │
│      SystemMessage(...),                            │
│      HumanMessage(...),                             │
│      AIMessage(                                     │
│        content="",                                  │
│        tool_calls=[                                 │
│          {name: "get_current_weather_summary"}      │
│        ]                                            │
│      )                                              │
│    ],                                               │
│    ...                                              │
│  }                                                  │
└────────────────┬────────────────────────────────────┘
                 │
                 │ Tools execute
                 ▼
┌─────────────────────────────────────────────────────┐
│  State After Tool Execution                         │
│                                                     │
│  {                                                  │
│    messages: [                                      │
│      ...,                                           │
│      ToolMessage(                                   │
│        content="🌤️ Today: 12-18°C, ..."             │
│      )                                              │
│    ],                                               │
│    weather_context: {                               │
│      temperature_min: 12,                           │
│      temperature_max: 18,                           │
│      ...                                            │
│    }                                                │
│  }                                                  │
└────────────────┬────────────────────────────────────┘
                 │
                 │ LLM formats final response
                 ▼
┌─────────────────────────────────────────────────────┐
│  Final State                                        │
│                                                     │
│  {                                                  │
│    messages: [                                      │
│      ...,                                           │
│      AIMessage(                                     │
│        content="Today in Lisbon: sunny, 12-18°C..." │
│      )                                              │
│    ],                                               │
│    weather_context: {...},                          │
│    ...                                              │
│  }                                                  │
└─────────────────────────────────────────────────────┘
```

---

## 7. Error Handling Flow

```
Tool Call: get_metro_status()
    │
    ▼
HTTP Request to Metro API
    │
    ├─► SUCCESS ──────────┐
    │                     │
    └─► TIMEOUT           │
         │                │
         ▼                │
    Retry #1 (wait 1s)    │
         │                │
         ├─► SUCCESS ─────┤
         │                │
         └─► TIMEOUT      │
              │           │
              ▼           │
         Retry #2 (2s)    │
              │           │
              ├─► SUCCESS ┤
              │           │
              └─► TIMEOUT │
                   │      │
                   ▼      │
            Retry #3 (4s) │
                   │      │
                   ├─►────┤
                   │      │
                   └─► FAIL
                        │
                        ▼
               ┌───────────────────┐
               │  Return Error Msg │
               │  to Agent         │
               └────────┬──────────┘
                        │
                        ▼
               ┌───────────────────┐
               │ Agent Formats     │
               │ User-Friendly     │
               │ Error Message     │
               └────────┬──────────┘
                        │
                        ▼
           "❌ Metro status temporarily
            unavailable. Try again in a
            few minutes."
```

---

## 8. Deployment Architecture (GitHub Actions)

```
┌────────────────────────────────────────────────┐
│          GitHub Repository                     │
│                                                │
│  - Source Code                                 │
│  - Data Files (JSON)                           │
│  - Vector DB (ChromaDB files)                  │
└───────────────┬────────────────────────────────┘
                │
                │ Push Trigger / Scheduled Cron
                │
                ▼
┌────────────────────────────────────────────────┐
│          GitHub Actions Runners                │
└────────────────────────────────────────────────┘
    │                          │
    │ 2 AM UTC                 │ 3 AM UTC
    │                          │
    ▼                          ▼
┌──────────────┐       ┌──────────────────┐
│ Web Scraping │       │ Vector Store     │
│   Workflow   │       │  Sync Workflow   │
└──────┬───────┘       └────────┬─────────┘
       │                        │
       │ Updates JSON           │ Reads JSON
       │                        │ Updates ChromaDB
       ▼                        ▼
┌──────────────────────────────────────────┐
│     Git Commit & Push                    │
│                                          │
│  - events.json                           │
│  - places.json                           │
│  - ChromaDB files                        │
└──────────────────────────────────────────┘
                │
                │ Updated Files
                ▼
┌──────────────────────────────────────────┐
│    Production Deployment                 │
│    (Streamlit Cloud / Local)             │
│                                          │
│  - Pulls latest from GitHub              │
│  - Loads updated vector store            │
│  - Serves users                          │
└──────────────────────────────────────────┘
```

---

**Mermaid Diagram Exports Available**:  
These ASCII diagrams can be converted to Mermaid for visualization tools.

---

*Created: December 30, 2025*  
*Author: André Filipe Gomes Silvestre*
