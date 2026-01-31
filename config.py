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
        MODEL_PROVIDER (str): Current LLM provider ('openai', 'lmstudio', 'azure').
        TEMPERATURE (float): LLM temperature setting (0 = deterministic).
        LISBON_GLOBAL_ID (int): IPMA global location ID for Lisbon (1110600).

    Usage:
        from config import Config

        # Access data paths
        places_path = Config.PATH_VISIT_LISBOA_PLACES

        # Get API key
        api_key = Config.OPENAI_API_KEY

        # Get model name
        model = Config.OPENAI_MODEL_NAME
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
    PATH_DADOS_ABERTOS_METADATA = (
        DATA_COLLECTION_DIR / "webscraping" / "lisbon_datasets_clean.json"
    )

    # Path to Turismo de Lisboa PDF guide
    # Contains: comprehensive tourist information in Portuguese/English
    PATH_PDF_TEXT = DATA_COLLECTION_DIR / "docs" / "Guia_LxCard_Ing_Esp_Abril_2024.pdf"

    # =========================================================================
    # MODEL PROVIDER SELECTION
    # =========================================================================
    # Available options:
    #   - 'openai'   : OpenAI API (GPT models)
    #   - 'azure'    : Azure OpenAI Service
    #   - 'lmstudio' : Local server (OpenAI-compatible API on port 1234)
    MODEL_PROVIDER = "azure"
    # MODEL_PROVIDER = "lmstudio"

    # =========================================================================
    # API KEYS (Environment Variables)
    # =========================================================================
    # Keys are loaded from environment variables for security.
    # Set these in a .env file in the project root:
    #   OPENAI_API_KEY=your_key_here

    # OpenAI API key (required for 'openai' provider)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Azure OpenAI credentials (required for 'azure' provider)
    # Create Azure OpenAI resource at: https://portal.azure.com
    # Documentation: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle
    #
    # NOTE: With v1 API (August 2025+), api_version is NOT required.
    # The code uses /openai/v1/ endpoint which auto-updates automatically.
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

    # =========================================================================
    # MODEL NAMES
    # =========================================================================

    # OpenAI GPT model
    # Options: gpt-5.2, gpt-5.1, gpt-5, gpt-5-mini
    # Can be set via environment variable OPENAI_MODEL_NAME
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "")

    # LM Studio model (Local server on port 1234)
    # Set to match the model loaded in your local LM Studio instance
    # LMSTUDIO_MODEL_NAME = 'openai/gpt-oss-20b'
    LMSTUDIO_MODEL_NAME = "qwen/qwen3-4b-2507"
    # LMSTUDIO_MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"
    # LMSTUDIO_MODEL_NAME = "qwen/qwen3-8b"
    LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

    # Azure OpenAI deployment name (must match your Azure deployment)
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv(
        "AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-nano"
    )

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

    # Debug/Development Settings
    # Show raw markdown responses in terminal for debugging/copying
    SHOW_MARDKOWN_RESPONSE_IN_TERMINAL = (
        False  # Set to True to print AI responses to terminal
    )

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

    # =========================================================================
    # MULTI-AGENT SYSTEM CONFIGURATION
    # =========================================================================
    # Enable/disable multi-agent mode (False = use single-agent V1)
    USE_MULTI_AGENT = True

    # Agent model assignments by provider
    # Each provider has its own agent configuration
    # Format: "agent_name": {"provider": "...", "model": "...", "temperature": ...}
    #
    # Recommendation:
    #   - Supervisor/Planner/Researcher: Use powerful models (reasoning-heavy)
    #   - Weather/Transport: Use fast/light models (tool-calling only)

    # LM STUDIO CONFIGURATION (Local models)
    AGENT_MODELS_LMSTUDIO = {
        "supervisor": {
            "provider": "lmstudio",
            "model": "qwen/qwen3-4b-2507",
            "temperature": 0.1,
        },
        "weather": {
            "provider": "lmstudio",
            "model": "qwen/qwen3-4b-2507",
            "temperature": 0,
        },
        "transport": {
            "provider": "lmstudio",
            "model": "qwen/qwen3-4b-2507",
            "temperature": 0,
        },
        "researcher": {
            "provider": "lmstudio",
            "model": "qwen/qwen3-4b-2507",
            "temperature": 0,
        },
        "planner": {
            "provider": "lmstudio",
            "model": "qwen/qwen3-4b-2507",
            "temperature": 0.1,
        },
    }

    # AZURE OPENAI CONFIGURATION (Cloud models)
    # NOTE: o-series/reasoning models (gpt-5-nano, o1, o3, etc.) only support temperature=1
    AGENT_MODELS_AZURE = {
        "supervisor": {
            "provider": "azure",
            "model": "gpt-5-nano",
            "temperature": 1,
        },
        "weather": {
            "provider": "azure",
            "model": "gpt-5-nano",
            "temperature": 1,
        },
        "transport": {
            "provider": "azure",
            "model": "gpt-5-nano",
            "temperature": 1,
        },
        "researcher": {
            "provider": "azure",
            "model": "gpt-5-nano",
            "temperature": 1,
        },
        "planner": {
            "provider": "azure",
            "model": "gpt-5-nano",
            "temperature": 1,
        },
    }

    # OPENAI CONFIGURATION (Direct API)
    AGENT_MODELS_OPENAI = {
        "supervisor": {
            "provider": "openai",
            "model": "gpt-5-nano",
            "temperature": 0.1,
        },
        "weather": {
            "provider": "openai",
            "model": "gpt-5-nano",
            "temperature": 0,
        },
        "transport": {
            "provider": "openai",
            "model": "gpt-5-nano",
            "temperature": 0,
        },
        "researcher": {
            "provider": "openai",
            "model": "gpt-5-nano",
            "temperature": 0,
        },
        "planner": {
            "provider": "openai",
            "model": "gpt-5-nano",
            "temperature": 0.1,
        },
    }

    # Active agent models - selected based on MODEL_PROVIDER
    @classmethod
    def get_agent_models(cls):
        """Returns agent models configuration based on current MODEL_PROVIDER."""
        if cls.MODEL_PROVIDER == "azure":
            return cls.AGENT_MODELS_AZURE
        elif cls.MODEL_PROVIDER == "openai":
            return cls.AGENT_MODELS_OPENAI
        else:  # lmstudio or default
            return cls.AGENT_MODELS_LMSTUDIO

    # Backwards compatibility - AGENT_MODELS as a classmethod
    @classmethod
    def AGENT_MODELS(cls):
        return cls.get_agent_models()

    # Fallback model when agent-specific config not found - also provider-dependent
    @classmethod
    def get_default_agent_model(cls):
        """Returns default agent model based on current MODEL_PROVIDER."""
        if cls.MODEL_PROVIDER == "azure":
            # o-series/reasoning models only support temperature=1
            return {"provider": "azure", "model": "gpt-5-nano", "temperature": 1}
        elif cls.MODEL_PROVIDER == "openai":
            return {"provider": "openai", "model": "gpt-5-nano", "temperature": 0}
        else:  # lmstudio or default
            return {
                "provider": "lmstudio",
                "model": "qwen/qwen3-4b-2507",
                "temperature": 0,
            }

    # Backwards compatibility - DEFAULT_AGENT_MODEL as a classmethod
    @classmethod
    def DEFAULT_AGENT_MODEL(cls):
        return cls.get_default_agent_model()


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
        status = (
            "\033[1;32m✓ Exists\033[0m"
            if path.exists()
            else "\033[1;31m✗ Missing\033[0m"
        )
        print(f"   {name}: {status}")

    # Test API keys
    print("\n\033[1m🔑 API Keys:\033[0m")
    print(f"   OPENAI_API_KEY: {'✓ Set' if Config.OPENAI_API_KEY else '✗ Not set'}")
    print(
        f"   AZURE_OPENAI_API_KEY: {'✓ Set' if Config.AZURE_OPENAI_API_KEY else '✗ Not set'}"
    )
    print(
        f"   AZURE_OPENAI_ENDPOINT: {'✓ Set' if Config.AZURE_OPENAI_ENDPOINT else '✗ Not set'}"
    )

    # Test model settings
    print("\n\033[1m🤖 Model Settings:\033[0m")
    print(f"   Provider: {Config.MODEL_PROVIDER}")
    print(f"   OpenAI Model: {Config.OPENAI_MODEL_NAME}")
    print(f"   Azure OpenAI Deployment: {Config.AZURE_OPENAI_DEPLOYMENT_NAME}")
    print(f"   LM Studio Model: {Config.LMSTUDIO_MODEL_NAME}")
    print(f"   Embedding Model: {Config.EMBEDDING_MODEL_NAME}")
    print(f"   Temperature: {Config.TEMPERATURE}")

    # Test Lisbon parameters
    print("\n\033[1m🌍 Lisbon Parameters:\033[0m")
    print(f"   Global ID: {Config.LISBON_GLOBAL_ID}")
    print(f"   Latitude: {Config.LISBON_LATITUDE}")
    print(f"   Longitude: {Config.LISBON_LONGITUDE}")
    print(f"   Area Aviso: {Config.LISBON_AREA_AVISO}")

    print("\n\033[1;32m✅ Configuration loaded successfully!\033[0m")
