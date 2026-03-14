# 🔭 Future Enhancements

This document captures only the shortlist of enhancements that remain realistically feasible with the current repository architecture as of 2026-03.

> [!NOTE]
> Everything below is prospective. If a capability is not documented elsewhere as implemented, treat it as a roadmap item only.

## ✅ Prioritized Shortlist

| Priority | Enhancement | User value | Difficulty | Why it is viable now |
|---------:|-------------|------------|------------|----------------------|
| 1 | iCalendar export for itineraries | users can add a generated plan to Google Calendar or Apple Calendar in one step | Low | itinerary responses already contain structured time blocks that can be converted into `.ics` events without new external services |
| 2 | Interactive itinerary map MVP | users can see ordered stops and the rough flow of the day instead of reading only text | Medium | the project already stores or retrieves many place coordinates, and a lightweight map layer can be added in Streamlit without a full routing engine |
| 3 | Downloadable itinerary pack | users can export a clean PDF or Markdown summary to keep offline during the trip | Low to Medium | the final answers are already Markdown-first, so export can reuse the existing response structure rather than inventing a new content pipeline |

## 1. iCalendar Export for Itineraries

### What it would do

- generate a `.ics` file from a finalized itinerary
- create one calendar entry per confirmed itinerary block
- work with Google Calendar, Apple Calendar, and Outlook through file import

### Why it is a strong next step

- it is immediately useful for both tourists and residents
- it does not require OAuth, background jobs, or third-party write permissions
- it keeps the system grounded because only confirmed itinerary items should be exported

### Implementation notes

- map planner time blocks into calendar events
- include title, start time, end time, place, and short notes when present
- skip blocks that do not have a reliable time window

## 2. Interactive Itinerary Map MVP

### What it would do

- plot itinerary stops as ordered pins
- show a simple visual sequence for the day
- optionally group segments by mode, for example walking, metro, bus, or train

### Why it is viable

- the repository already works with coordinates from VisitLisboa, Lisboa Aberta, Metro, Carris, and Carris Metropolitana contexts
- a first useful version can be built with a lightweight map component instead of a full navigation engine
- it improves readability without changing the agent architecture

### Scope guardrails

- this should be an MVP map, not a professional routing engine
- do not promise turn-by-turn navigation
- do not block the feature on perfect coordinates for every single stop

## 3. Downloadable Itinerary Pack

### What it would do

- let the user download the final itinerary as PDF or Markdown
- preserve sections like timetable, transport notes, weather caveats, and source attribution
- support offline use during the visit

### Why it is viable

- the app already renders structured Markdown responses
- export can reuse the final planner output after formatting
- the implementation does not depend on new APIs or provider-side features

## 🚫 Not Prioritized in This Shortlist

The following ideas were intentionally left out of the shortlist because they push effort or maintenance too far beyond the requested low or medium range:

- direct calendar write integrations through OAuth
- a full-featured professional map with advanced routing and live replanning engine behavior
- large new transport-provider integrations that would expand operational scope more than user value in the short term

## 📌 Decision Rule for New Roadmap Items

Any new enhancement should pass all four checks before being added here:

1. it must provide immediate user value
2. it must be implementable with the current architecture or with a small extension
3. it must stay in the low or medium difficulty range
4. it must not create major documentation or operational drift around the supported `app.py` path
