# ==========================================================================
# Master Thesis - Configuration
#   - André Filipe Gomes Silvestre, 2025
# 
#   Central configuration for paths, model selection, and API keys.
#   Updated to support Cloud APIs (Google, OpenAI, Groq) alongside Local.
# ==========================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).parent.resolve()


class Config:
    """
    Global configuration settings.
    """
    # ---------------------------------------------------------
    # 📂 DATA PATHS
    # ---------------------------------------------------------
    DATA_COLLECTION_DIR = BASE_DIR / "data_collection"
    VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"
    
    PATH_VISIT_LISBOA_PLACES = DATA_COLLECTION_DIR / "webscraping" / "places.json"
    PATH_VISIT_LISBOA_EVENTS = DATA_COLLECTION_DIR / "webscraping" / "events.json"
    PATH_DADOS_ABERTOS_METADATA = DATA_COLLECTION_DIR / "webscraping" / "lisbon_datasets.json"
    PATH_PDF_TEXT = DATA_COLLECTION_DIR / "docs" / "Guia_LxCard_Ing_Esp_Abril_2024.pdf"
    
    # ---------------------------------------------------------
    # 🤖 MODEL SELECTION
    # ---------------------------------------------------------
    # Options: 'google', 'openai', 'groq' (for Qwen), 'ollama' (local)
    MODEL_PROVIDER = "groq"  # <--- CHANGE THIS TO SWITCH PROVIDER
    
    # ---------------------------------------------------------
    # 🔑 API KEYS (Load from Environment for security)
    # ---------------------------------------------------------
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Required for OpenAI models
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Required for Gemma/Gemini
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")     # Required for Qwen (Cloud API)

    # ---------------------------------------------------------
    # 📝 MODEL NAMES
    # ---------------------------------------------------------
    
    # 1. Google (Free Tier available via AI Studio)
    # Note: If 'gemma-3-27b' is not yet released via API, usually 'gemini-pro' or 'gemma-2-27b-it' is used.
    # We keep the name you requested as a placeholder.
    GOOGLE_MODEL_NAME = "gemini-2.0-flash-exp"  # Or "gemma-3-27b-it" when available
    
    # 2. OpenAI 
    # Placeholder for the "gpt-oss-20b" you mentioned. 
    # Ensure this model name is valid for your specific OpenAI access/proxy.
    OPENAI_MODEL_NAME = "gpt-oss-20b" # Or standard "gpt-4o-mini"
    
    # 3. Qwen (via Groq Cloud API)
    # Groq hosts Qwen models with extremely fast inference.
    GROQ_MODEL_NAME = "qwen/qwen3-32b" # Placeholder for ""
    
    # 4. Local Model
    LOCAL_MODEL_NAME = "model_name" 

    # ---------------------------------------------------------
    # 🧠 MEMORY & EMBEDDINGS
    # ---------------------------------------------------------
    # Embedding Model (HuggingFace)
    # Runs locally on CPU/GPU. Safe choice for thesis.
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3" 
    
    # Temperature for reasoning (0 = deterministic)
    TEMPERATURE = 0
