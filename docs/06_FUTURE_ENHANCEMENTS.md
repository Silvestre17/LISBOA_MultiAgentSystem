# 🔭 Future Enhancements

This document captures likely next steps for LISBOA without presenting them as already implemented.

> [!NOTE]
> Everything below is prospective. If a capability is not documented elsewhere as implemented, treat it as a roadmap item only.

## 🧭 Roadmap Themes

| Theme | Why it matters |
|------|----------------|
| richer UX artefacts | improve readability and planning usefulness for end users |
| stronger agent coordination | make follow-up interactions and constraints more reliable |
| better retrieval quality | improve grounding, filtering, and fallback behavior |
| broader transport intelligence | support more providers and better disruption handling |
| stronger evaluation depth | make thesis claims more robust and reproducible |
| tighter operational hygiene | reduce documentation drift and recurring maintenance overhead |

## 🎨 Product and UX

Potential next steps:

- optional place images in itinerary responses
- lightweight static route maps for complex multimodal plans
- richer place cards for attractions, services, and events
- clearer mobile-first presentation patterns in the Streamlit UI

### Single Supported UI Evolution

The public documentation already treats `app.py` as the supported Streamlit entrypoint. Future UI work should continue consolidating improvements into that single documented entrypoint instead of expanding the number of public-facing application variants.

## 🤖 Agent and Orchestration Improvements

Potential next steps:

- stronger memory handling for follow-up planning sessions
- richer planner constraints such as budget envelopes or tighter accessibility preferences
- more explicit source-citation policies in final answers
- broader deterministic QA coverage across non-transport domains
- clearer retry policies when QA detects partial worker outputs

## 📚 Retrieval and Knowledge Improvements

Potential next steps:

- richer metadata filters for event date, region, and venue
- improved hybrid retrieval strategies where dense retrieval is complemented by lexical signals
- clearer fallback behavior when semantic retrieval returns sparse results
- more structured image and schedule metadata in the vector store
- better differentiation between tourism content and resident-oriented service discovery

## 🚇 Transport Improvements

Potential next steps:

- additional operator coverage within the AML ecosystem
- better disruption-aware itinerary replanning
- route-map artefacts generated from multimodal outputs
- stronger ETA reasoning for live vehicle tracking outputs
- clearer confidence or freshness cues in transport-heavy answers

## 🧪 Evaluation Improvements

Potential next steps:

- larger benchmark sets beyond the current 72-query dataset
- broader human calibration and inter-rater agreement analysis
- longitudinal evaluation across seasons or service disruptions
- stronger domain-specific validators outside transport
- more systematic export and visualization workflows for result comparison

## ⚙️ Operations and Maintainability

Potential next steps:

- more compact documentation around recurring operations
- changelog-style tracking for tool inventory changes
- stronger CI verification for documentation drift
- automated checks for stale counts and workflow times in Markdown files
- clearer runbooks for regenerating evaluation artefacts and local transport support files

## ✅ Guardrails for Future Work

As the project evolves, a few principles should stay fixed:

- keep grounded data access explicit and visible in the architecture
- avoid documenting aspirational features as if they already exist
- update tests and documentation when tool inventory or workflows change
- prefer one clearly supported operating path over multiple loosely documented variants
