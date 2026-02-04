# Roadmap

This document captures planned enhancements.

## Multimodal capabilities (maps and images)

Goal: include map-friendly artifacts and optional images in itinerary outputs.

### Image integration

Potential approach:

- Extend VisitLisboa search tool outputs to include image URLs when available.
- Ensure the planner agent propagates those links into the final itinerary.
- Render as Markdown images or links depending on the UI.

Implementation notes:

- VisitLisboa scraping already collects `image_urls` for events and places.
- The semantic search layer can expose image URLs via metadata.
- Preferred output format for the UI is Markdown images: `![Title](url)`.

Concrete steps:

- Tool output: extend `tools/visitlisboa_api.py` tool responses to include a representative image URL when available (as a Markdown image or a link).
- Planning: ensure the planner agent propagates image links into the final itinerary in a controlled way (images should remain optional and limited in count).
- UI: Streamlit Markdown rendering already supports images, so this should not require UI changes beyond layout tuning.

### Map generation

Potential approach:

- Add a map rendering tool that can output a static PNG for a route.
- Integrate it optionally when routes are complex.

Library candidates:

- `staticmap` (generate a PNG image from OSM tiles)
- `folium` (interactive HTML maps, best when the UI can render HTML)

MVP idea:

- Add a new tool (for example `tools/map_renderer.py`) that accepts coordinates and optional waypoints.
- Save route images under a local folder and return a file path or URL that the UI can render.

Design notes:

- Prefer static PNG output for robustness in chat style UIs.
- For an MVP, a straight line between origin and destination is acceptable, but a later iteration should use route polylines from a routing backend.

Integration point:

- The multimodal routing tool (`get_route_between_stations`) can optionally trigger map generation for complex routes and append the resulting image path or URL.

Example (target UX):

- A route answer can include both steps and a rendered artifact, for example:
	- Steps: "Take metro to Rossio, then walk 10 minutes..."
	- Map: `![Route map](path-or-url)`
	- Place card: `![Place image](image-url)`

Notes:

- Keep the map layer optional so the core itinerary planner remains fast and robust.
