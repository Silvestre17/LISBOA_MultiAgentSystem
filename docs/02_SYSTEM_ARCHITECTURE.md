# 🏗️ LISBOA System Architecture

This document describes the runtime architecture implemented in the repository today. The supported default path is the multi-agent flow inside `MultiAgentAssistant` in `agent/graph.py`.

> [!IMPORTANT]
> The repository still contains the single-agent `LisbonAssistant` for compatibility, but the documented default runtime is the multi-agent system.

## 🖼️ Conceptual Framework Figure

The latest thesis-facing framework figure stored in the repository is shown below.

![LISBOA framework figure](../img/LISBOA_Framework_fev2026.png)


## 🧩 High-Level Components

| Layer | Main files | Responsibility |
|------|------------|----------------|
| UI | `app.py` | Streamlit chat interface, provider selection, session state, quick actions |
| Orchestration | `agent/graph.py` | routing, parallel worker execution, QA pass, final response assembly |
| State | `agent/state.py` | shared `AgentState` and user-context schema |
| LLM provider factory | `agent/llm_factory.py` | provider-specific model creation and binding |
| Specialized agents | `agent/agents/` | domain routing, retrieval, validation, synthesis |
| Tool and data layer | `tools/` | live APIs, open data access, vector search, support utilities |

## 🧱 Architecture Layers

### 🎨 UI Layer

- Streamlit chat experience through `app.py`
- runtime provider and model selection
- session state, quick actions, info pages, and status updates
- pre-warming of the vector store and Carris support database at startup

### 🤖 Orchestration Layer

- `SupervisorAgent` classifies the query and decides which workers to call
- worker agents execute domain-specific retrieval
- `QualityAssuranceAgent` validates completeness and factual consistency
- `PlannerAgent` synthesizes itinerary-style answers when planning is required

### 🔌 Tool and Data Layer

- live APIs for weather and transport
- VisitLisboa semantic retrieval over ChromaDB
- Lisboa Aberta on-demand geospatial discovery
- web fallback for history and culture

## 🔁 End-to-End Runtime Flow

```mermaid
graph TD
  User([User]) --> UI[app.py]
  UI --> Supervisor[SupervisorAgent]

  Supervisor -->|direct response| UI
  Supervisor -->|weather| Weather[WeatherAgent]
  Supervisor -->|transport| Transport[TransportAgent]
  Supervisor -->|research| Researcher[ResearcherAgent]

  Weather --> QA[QualityAssuranceAgent]
  Transport --> QA
  Researcher --> QA

  QA -->|planning query| Planner[PlannerAgent]
  QA -->|single or combined response| UI
  Planner --> UI
```

### What this means in practice

1. The user sends a request through `app.py`.
2. `SupervisorAgent.route()` decides whether the answer can be returned directly or whether worker agents are needed.
3. Weather, transport, and research workers can run in parallel.
4. `QualityAssuranceAgent.validate()` checks completeness, disclaimers, and retry needs.
5. If the route includes planning, `PlannerAgent.synthesize()` writes the final itinerary.
6. Otherwise, the system returns a direct or combined response without invoking the planner.

## 🤝 Agent Roles and Tool Assignments

| Agent | Primary role | Assigned tools | Notes |
|------|--------------|---------------:|------|
| `SupervisorAgent` | query routing and direct handling | 0 | returns direct responses for greetings and out-of-scope cases |
| `WeatherAgent` | weather retrieval | 4 | IPMA only |
| `TransportAgent` | transport retrieval | 30 | Metro, Carris Metropolitana, Carris Urban, CP, and multimodal tools |
| `ResearcherAgent` | tourism, services, and knowledge retrieval | 11 | VisitLisboa, Lisboa Aberta, and web fallback |
| `QualityAssuranceAgent` | validation and retry guidance | 0 | validates outputs, adds disclaimers, and can trigger one retry path |
| `PlannerAgent` | final planning synthesis | 0 | only used when the route explicitly includes the planner |

## 🎯 Final Response Semantics

One of the most important architectural details is that the planner is **not** always the final responder.

- **Planning queries:** the final answer is synthesized by `PlannerAgent.synthesize()`.
- **Greetings or out-of-scope requests:** the response can be returned directly by the supervisor.
- **Simple single-domain requests:** the response can come from one specialist worker or from combined worker outputs, without using the planner.

This behavior is implemented in `MultiAgentAssistant.chat()` in `agent/graph.py`.

## ✅ QA in the Real Runtime

`QualityAssuranceAgent` is not a general conversational front-end. Its runtime role is to:

- inspect worker outputs
- detect missing critical data
- attach disclaimers about data limitations
- guide a single retry path when worker outputs are incomplete
- perform deterministic validation for certain factual checks

The QA step happens **after** worker execution and **before** final synthesis or response combination.

## 🛠️ Tool Mapping by Worker

| Worker | Composition |
|--------|-------------|
| `WeatherAgent` | IPMA, 4 tools |
| `TransportAgent` | Metro 6 + Carris Metropolitana 8 + Carris Urban 8 + CP 6 + multimodal 2 |
| `ResearcherAgent` | VisitLisboa 5 + Lisboa Aberta 5 + web knowledge 1 |

## 🧠 State Management

`agent/state.py` defines `AgentState`, which carries:

| State field | Purpose |
|-------------|---------|
| `messages` | conversation history |
| `user_context` | language, location, mobility, preferences, available time |
| `weather_context` | cached weather context when available |
| `transport_context` | cached transport context when available |
| `current_plan` | current itinerary structure |
| `candidate_pois` | retrieved places under consideration |
| `events_data` | retrieved event records for planning |
| `agents_to_call` | supervisor routing decision |
| `agent_outputs` | collected worker outputs |
| `iteration_count` | loop-prevention and execution-tracking metadata |

## ⚙️ Provider and Model Selection

`agent/llm_factory.py` supports the following provider families:

- **LM Studio**
- **OpenAI**
- **Azure OpenAI**

Per-agent model selection is controlled in `config.py` through `AGENT_MODELS_LMSTUDIO`, `AGENT_MODELS_OPENAI`, and `AGENT_MODELS_AZURE`. The Streamlit sidebar in `app.py` can override provider-level and per-agent model choices at runtime.

## 🛡️ Reliability and Control Mechanisms

The implemented architecture includes:

- loop detection for repeated tool calls
- safe LLM invocation with Azure content-filter retry handling
- parallel worker execution with context propagation
- a single QA-guided retry path when required
- response cleanup and formatting before Streamlit rendering
- deterministic validation in the QA stage
- per-agent usage and latency tracking hooks
