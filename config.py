# ==========================================================================
# Master Thesis - Configuration
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Central configuration file for the Lisbon Urban Assistant project.
#   Manages all settings including:
#     - File paths for data sources
#     - LLM provider and model selection
#     - API keys (loaded from environment)
#     - Embedding model configuration
#     - Lisbon-specific parameters for IPMA API
# 
#   Security Note:
#     API keys are loaded from environment variables (.env file).
#     Never commit actual API keys to version control.
# ==========================================================================

# Required libraries:
# pip install python-dotenv

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
# This allows secure storage of API keys outside the codebase
load_dotenv()

# Base directory: the root folder of this project
BASE_DIR = Path(__file__).parent.resolve()


class Config:
    """
    Global configuration settings for the Lisbon Urban Assistant.
    
    This class provides centralized access to all configuration parameters
    used throughout the application, including paths, model settings, and API keys.
    
    Attributes:
        DATA_COLLECTION_DIR (Path): Directory containing collected data sources.
        VECTOR_DB_DIR (Path): Directory for ChromaDB vector store persistence.
        MODEL_PROVIDER (str): Current LLM provider ('groq', 'google', 'openai', 'lmstudio', 'ollama').
        TEMPERATURE (float): LLM temperature setting (0 = deterministic).
        LISBON_GLOBAL_ID (int): IPMA global location ID for Lisbon (1110600).
    
    Usage:
        from config import Config
        
        # Access data paths
        places_path = Config.PATH_VISIT_LISBOA_PLACES
        
        # Get API key
        api_key = Config.GROQ_API_KEY
        
        # Get model name
        model = Config.GROQ_MODEL_NAME
    """
    
    # =========================================================================
    # DATA PATHS
    # =========================================================================
    # Directory containing all data collection scripts and outputs
    DATA_COLLECTION_DIR = BASE_DIR / "data_collection"
    
    # Directory for ChromaDB vector store (persistent embeddings)
    VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"
    
    # Path to VisitLisboa places JSON (webscraping output)
    # Contains: museums, restaurants, attractions, etc.
    PATH_VISIT_LISBOA_PLACES = DATA_COLLECTION_DIR / "webscraping" / "places.json"
    
    # Path to VisitLisboa events JSON (webscraping output)
    # Contains: cultural events, festivals, exhibitions, etc.
    PATH_VISIT_LISBOA_EVENTS = DATA_COLLECTION_DIR / "webscraping" / "events.json"
    
    # Path to Dados Abertos metadata JSON (webscraping output)
    # Contains: 310+ GeoJSON datasets from CMLisboa open data portal
    PATH_DADOS_ABERTOS_METADATA = DATA_COLLECTION_DIR / "webscraping" / "lisbon_datasets_clean.json"
    
    # Path to Turismo de Lisboa PDF guide
    # Contains: comprehensive tourist information in Portuguese/English
    PATH_PDF_TEXT = DATA_COLLECTION_DIR / "docs" / "Guia_LxCard_Ing_Esp_Abril_2024.pdf"
    
    # =========================================================================
    # MODEL PROVIDER SELECTION
    # =========================================================================
    # Available options:
    #   - 'groq'     : Cloud API with high-speed inference (Qwen, Llama)
    #   - 'google'   : Google AI Studio (Gemini models)
    #   - 'openai'   : OpenAI API (GPT models)
    #   - 'lmstudio' : Local server (OpenAI-compatible API on port 1234)
    #   - 'ollama'   : Local Ollama server (various open models)
    MODEL_PROVIDER = "lmstudio"  # <-- Change this to switch providers
    
    # =========================================================================
    # API KEYS (Environment Variables)
    # =========================================================================
    # Keys are loaded from environment variables for security.
    # Set these in a .env file in the project root:
    #   OPENAI_API_KEY=your_key_here
    #   GOOGLE_API_KEY=your_key_here
    #   GROQ_API_KEY=your_key_here
    
    # OpenAI API key (required for 'openai' provider)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Google API key (required for 'google' provider)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Groq API key (required for 'groq' provider)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # =========================================================================
    # MODEL NAMES
    # =========================================================================
    
    # Google Gemini model
    # Options: gemini-3-pro, gemini-3-flash, gemini-2.5-pro, gemini-2.5-flash e gemini-2.5-flash-lite
    GOOGLE_MODEL_NAME = "gemini-3-flash"
    
    # OpenAI GPT model
    # Options: gpt-5.2, gpt-5.1, gpt-5, gpt-5-mini
    OPENAI_MODEL_NAME = "gpt-5-mini"
    
    # Groq model (High-speed inference for open models)
    # Available models for tool calling (as of Dec 2025):
    #   - openai/gpt-oss-120b AND openai/gpt-oss-20b
    #   - meta-llama/llama-4-scout-17b-16e-instruct
    #   - qwen/qwen3-32b
    #   - moonshotai/kimi-k2-instruct-0905
    # Check available models: https://console.groq.com/docs/models
    GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
    
    # LM Studio model (Local server on port 1234)
    # Set to match the model loaded in your local LM Studio instance
    # LMSTUDIO_MODEL_NAME = 'openai/gpt-oss-20b'
    LMSTUDIO_MODEL_NAME = "qwen/qwen3-4b-2507"
    LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
    
    # Ollama model (Local model execution)
    # Requires: ollama pull qwen2.5:7b && ollama serve
    # OLLAMA_MODEL_NAME = "qwen/qwen2.5:7b"
    OLLAMA_MODEL_NAME = "openai/gpt-oss-20b"
    
    # =========================================================================
    # EMBEDDINGS CONFIGURATION
    # =========================================================================
    # Embedding model for vector store (runs locally via HuggingFace)
    # BAAI/bge-m3 is multilingual and works well with Portuguese/English
    # Alternatives: sentence-transformers/all-MiniLM-L6-v2 (faster, English-only)
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
    
    # =========================================================================
    # LLM PARAMETERS
    # =========================================================================
    # Temperature controls randomness in LLM responses
    # 0 = deterministic (same input -> same output)
    # 0.7 = moderate creativity
    # 1.0 = high creativity/randomness
    TEMPERATURE = 0
    
    # =========================================================================
    # LISBON GEOGRAPHIC PARAMETERS (IPMA)
    # =========================================================================
    # These are used for IPMA weather API calls
    # Source: https://api.ipma.pt/open-data/distrits-islands.json
    
    # IPMA Global ID for Lisbon municipality
    LISBON_GLOBAL_ID = 1110600
    
    # Geographic coordinates (WGS84)
    LISBON_LATITUDE = 38.7660
    LISBON_LONGITUDE = -9.1286
    
    # Area code for weather warnings (LSB = Lisbon district)
    LISBON_AREA_AVISO = "LSB"


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    """
    Test script to verify configuration loading.
    
    Expected Output:
        - All paths should exist or be valid
        - API keys should be loaded (or show 'Not set')
        - Model names should be displayed
        - Lisbon parameters should be correct
    """
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m📋 Configuration Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    # Test paths
    print("\n\033[1m📂 Data Paths:\033[0m")
    print(f"   Base Directory: {BASE_DIR}")
    print(f"   Data Collection: {Config.DATA_COLLECTION_DIR}")
    print(f"   Vector DB: {Config.VECTOR_DB_DIR}")
    print(f"   Places JSON: {Config.PATH_VISIT_LISBOA_PLACES}")
    print(f"   Events JSON: {Config.PATH_VISIT_LISBOA_EVENTS}")
    print(f"   Datasets JSON: {Config.PATH_DADOS_ABERTOS_METADATA}")
    
    # Check if paths exist
    print("\n\033[1m✅ Path Validation:\033[0m")
    paths_to_check = [
        ("Places JSON", Config.PATH_VISIT_LISBOA_PLACES),
        ("Events JSON", Config.PATH_VISIT_LISBOA_EVENTS),
        ("Datasets JSON", Config.PATH_DADOS_ABERTOS_METADATA),
    ]
    for name, path in paths_to_check:
        status = "\033[1;32m✓ Exists\033[0m" if path.exists() else "\033[1;31m✗ Missing\033[0m"
        print(f"   {name}: {status}")
    
    # Test API keys
    print("\n\033[1m🔑 API Keys:\033[0m")
    print(f"   GROQ_API_KEY: {'✓ Set' if Config.GROQ_API_KEY else '✗ Not set'}")
    print(f"   GOOGLE_API_KEY: {'✓ Set' if Config.GOOGLE_API_KEY else '✗ Not set'}")
    print(f"   OPENAI_API_KEY: {'✓ Set' if Config.OPENAI_API_KEY else '✗ Not set'}")
    
    # Test model settings
    print("\n\033[1m🤖 Model Settings:\033[0m")
    print(f"   Provider: {Config.MODEL_PROVIDER}")
    print(f"   Groq Model: {Config.GROQ_MODEL_NAME}")
    print(f"   Google Model: {Config.GOOGLE_MODEL_NAME}")
    print(f"   OpenAI Model: {Config.OPENAI_MODEL_NAME}")
    print(f"   LM Studio Model: {Config.LMSTUDIO_MODEL_NAME}")
    print(f"   Ollama Model: {Config.OLLAMA_MODEL_NAME}")
    print(f"   Embedding Model: {Config.EMBEDDING_MODEL_NAME}")
    print(f"   Temperature: {Config.TEMPERATURE}")
    
    # Test Lisbon parameters
    print("\n\033[1m🌍 Lisbon Parameters:\033[0m")
    print(f"   Global ID: {Config.LISBON_GLOBAL_ID}")
    print(f"   Latitude: {Config.LISBON_LATITUDE}")
    print(f"   Longitude: {Config.LISBON_LONGITUDE}")
    print(f"   Area Aviso: {Config.LISBON_AREA_AVISO}")
    
    print("\n\033[1;32m✅ Configuration loaded successfully!\033[0m")
