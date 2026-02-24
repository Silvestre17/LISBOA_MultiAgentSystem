# Lisbon Urban Assistant Documentation

Master Thesis: LISBOA - Lisbon Itenerary System Based On AI (NOVA IMS, 2025 to 2026)

This documentation uses a flat, English-only structure that matches the current repository state.

## Prerequisites

### System Requirements
- **Python:** 3.11 or higher
- **Operating Systems:** Windows 10/11, Ubuntu 20.04+, macOS 12+
- **RAM:** 8GB minimum (16GB recommended for vector DB sync)
- **Disk Space:** 2GB for dependencies, 500MB for vector database

### Optional Requirements
- **LM Studio:** For local LLM inference (recommended for development)
- **GPU:** CUDA-compatible GPU for faster embeddings (optional)

### Health Check
```bash
python -c "import chromadb; import langchain; import streamlit; print('✅ All dependencies OK')"
```

## Quick start

1. Install dependencies: `pip install -r requirements.txt`
2. Configure environment variables in a local `.env` file (see `docs/OPERATIONS.md`)
3. Build or update the vector store (first time): `python tools/vector_store.py`
4. Run the Streamlit UI: `streamlit run app.py`

## Documentation map

### Project Overview

- Complete project introduction: `docs/01_PROJECT_OVERVIEW.md`

### Tools

- Tools, APIs, examples, and testing: `docs/03_TOOLS_REFERENCE.md`

### Architecture

- System and agent architecture: `docs/02_SYSTEM_ARCHITECTURE.md`

### Data

- Datasets, scraping outputs, vector DB: `docs/04_DATA_SOURCES_AND_SCHEMAS.md`

### Operations

- Environment, GitHub Actions, performance: `docs/05_DEPLOYMENT_AND_OPERATIONS.md`

### Roadmap

- Planned multimodal extensions: `docs/06_FUTURE_ENHANCEMENTS.md`

## Notes on output language

Some tool outputs contain Portuguese strings because the system targets Lisbon and the AML region. Documentation is English-only, but it does not change the current tool output strings.

## Minimal .env example

```bash
# LLM providers
OPENAI_API_KEY=...
OPENAI_MODEL_NAME=gpt-5-nano

AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5-nano

# Metro (official API, optional)
METRO_CONSUMER_KEY=...
METRO_CONSUMER_SECRET=...

# Web knowledge (optional)
TAVILY_API_KEY=...

# LangSmith tracing (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=...
```

Last updated: 2026-02-04 (Documentation restructured with uniform naming convention)