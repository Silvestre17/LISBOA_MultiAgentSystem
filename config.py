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
#     API keys are loaded from environment variables: Streamlit secrets
#     (root-level keys), the process environment and the .env file.
#     Never commit actual API keys to version control.
# ==========================================================================

# Required libraries:
# pip install python-dotenv

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from tools.runtime_paths import get_runtime_data_root, is_hosted_deployment


def _load_streamlit_secrets() -> None:
    """Expose root-level Streamlit secrets as environment variables.

    ``streamlit run`` parses ``secrets.toml`` when the server starts and copies
    root-level values into ``os.environ`` (Streamlit Community Cloud provides
    the Secrets panel as that file). Loading again here is a cached no-op in
    that case and guarantees the values exist before the settings below are
    read, whatever started the server. Scripts, evaluation runs and CI do not
    run a Streamlit server and load nothing.
    """
    streamlit_module = sys.modules.get("streamlit")
    if streamlit_module is None:
        return
    from streamlit import runtime

    if runtime.exists():
        streamlit_module.secrets.load_if_toml_exists()


# Precedence: Streamlit secrets, then the process environment, then .env
# (load_dotenv never overrides a variable that is already set).
_load_streamlit_secrets()
load_dotenv()

__all__ = ["Config"]

# Base directory: the root folder of this project
BASE_DIR = Path(__file__).parent.resolve()


def _env_bool(name: str, default: bool = False) -> bool:
    """Return a boolean setting from an environment variable.

    Accepted truthy values are case-insensitive: ``true``, ``1``, ``yes``,
    ``y``, and ``on``. Accepted falsey values are ``false``, ``0``, ``no``,
    ``n``, and ``off``. Any missing or unrecognized value falls back to the
    provided default.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    return default


def _env_positive_int_or_none(name: str, default: int | None = None) -> int | None:
    """Return a positive integer setting, or ``None`` when unset/invalid."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = int(raw_value.strip())
    except ValueError:
        return default

    return value if value > 0 else default


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

    # Hugging Face Space, Streamlit Community Cloud, or LISBOA_HOSTED=true.
    # See tools/runtime_paths.py for the detection rules.
    IS_HOSTED = is_hosted_deployment()

    # Writable runtime data root for generated databases: data/ locally, the
    # system temporary directory on hosted deployments (LISBOA_RUNTIME_DATA_DIR
    # overrides both).
    RUNTIME_DATA_DIR = get_runtime_data_root()

    # Directory for ChromaDB vector store (persistent embeddings)
    VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", str(RUNTIME_DATA_DIR / "vector_db")))

    # Runtime vector database artifact download.
    # Hosted deployments should keep data/vector_db out of the Space git history
    # and fetch the latest release asset at startup when the local ChromaDB
    # files are missing.
    VECTOR_DB_RELEASE_ENABLED = _env_bool("VECTOR_DB_RELEASE_ENABLED", True)
    VECTOR_DB_RELEASE_REPO = os.getenv("VECTOR_DB_RELEASE_REPO", "Silvestre17/LISBOA_MultiAgentSystem")
    VECTOR_DB_RELEASE_TAG = os.getenv("VECTOR_DB_RELEASE_TAG", "vector-db-latest")
    VECTOR_DB_RELEASE_ASSET = os.getenv("VECTOR_DB_RELEASE_ASSET", "vector_db.zip")
    VECTOR_DB_RELEASE_TOKEN = (
        os.getenv("VECTOR_DB_RELEASE_TOKEN")
        or os.getenv("GITHUB_RELEASE_TOKEN")
        or os.getenv("GITHUB_TOKEN")
    )
    VECTOR_DB_RELEASE_FORCE_DOWNLOAD = _env_bool("VECTOR_DB_RELEASE_FORCE_DOWNLOAD", False)
    VECTOR_DB_RELEASE_TIMEOUT_SECONDS = int(os.getenv("VECTOR_DB_RELEASE_TIMEOUT_SECONDS", "120"))

    # Hosted deployments stay up for days while the workflows republish the
    # vector DB daily and the operators update their GTFS feeds. At most once
    # per interval, a request starts a background check that swaps in newer
    # data without restarting the app.
    RUNTIME_DATA_REFRESH_SECONDS = 3600 * _env_positive_int_or_none("LISBOA_DATA_REFRESH_HOURS", 6)

    # Versioned local LLM pricing snapshot used for runtime/eval cost accounting.
    # This avoids live pricing lookups during user requests and keeps summaries
    # reproducible for a known snapshot date.
    LLM_PRICING_CATALOG_PATH = BASE_DIR / "data" / "pricing" / "llm_model_pricing.json"

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
    # Production deploy keeps a single provider active.
    # Available options:
    #   - 'azure'    : Azure OpenAI Service
    #   - 'openai'   : OpenAI API (GPT models)
    #   - 'lmstudio' : Local server (OpenAI-compatible API on port 1234)
    MODEL_PROVIDER = "azure"
    # MODEL_PROVIDER = "openai"
    # MODEL_PROVIDER = "lmstudio"

    # Visitor model settings in the sidebar.
    # When True, each visitor can switch their own browser session to any
    # OpenAI-compatible endpoint with their own key and model. The key stays in
    # that Streamlit session and is never written to the environment, this
    # class, files or traces. The server provider above remains the default.
    ALLOW_VISITOR_LLM_SETTINGS = _env_bool("LISBOA_ALLOW_VISITOR_LLM_SETTINGS", True)

    # Whether a visitor base URL may resolve to a loopback, private, link-local,
    # reserved or multicast address (local servers such as LM Studio). Off by
    # default on hosted deployments, so visitors cannot make the server call
    # its own internal network; there, every visitor request is also re-checked
    # at connection time and redirects are not followed (agent/llm_factory.py).
    ALLOW_PRIVATE_LLM_HOSTS = _env_bool("LISBOA_ALLOW_PRIVATE_LLM_HOSTS", not IS_HOSTED)

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
    # .NOTE: With v1 API (August 2025+), api_version is NOT required.
    # The code uses /openai/v1/ endpoint which auto-updates automatically.
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

    # =========================================================================
    # MODEL NAMES
    # =========================================================================

    # OpenAI GPT model
    # Options: gpt-5.4-mini, gpt-5.4, gpt-5.2 (see https://platform.openai.com/docs/models)
    # Can be set via environment variable OPENAI_MODEL_NAME
    # Default is set to a valid model name to avoid runtime errors.
    # > SUPER IMPORTANT: Choose model
    DEFAULT_GPT_MODEL_NAME = "gpt-5.4-mini"
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", DEFAULT_GPT_MODEL_NAME)

    # LM Studio model (Local server on port 1234)
    # Set to match the model loaded in your local LM Studio instance
    # LMSTUDIO_MODEL_NAME = 'openai/gpt-oss-20b'
    LMSTUDIO_MODEL_NAME = "qwen/qwen3.5-9b"
    # LMSTUDIO_MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"
    # LMSTUDIO_MODEL_NAME = "qwen/qwen3-8b"
    LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

    # Azure OpenAI deployment name (must match your Azure deployment)
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv(
        "AZURE_OPENAI_DEPLOYMENT_NAME", DEFAULT_GPT_MODEL_NAME
    )

    # =========================================================================
    # EMBEDDINGS CONFIGURATION
    # =========================================================================
    # Embedding model of the vector store. BAAI/bge-m3 is multilingual and
    # works well with Portuguese/English. The ChromaDB store is always built
    # with the local model (L2-normalised vectors); queries can use either:
    #   - 'local'      : sentence-transformers on this machine; needs
    #                    requirements_local_embeddings.txt (torch, ~3-4 GB RAM)
    #   - 'cloudflare' : the same model on Cloudflare Workers AI; needs
    #                    CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN
    #   - 'auto'       : 'cloudflare' when both credentials are set, else 'local'
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
    EMBEDDINGS_BACKEND = os.getenv("LISBOA_EMBEDDINGS_BACKEND", "auto").strip().lower()
    CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    # Cloudflare API token with the "Workers AI: Read" permission only.
    CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
    CLOUDFLARE_EMBEDDING_MODEL = "@cf/baai/bge-m3"

    @classmethod
    def embeddings_backend(cls) -> str:
        """Return the query embeddings backend in use: ``local`` or ``cloudflare``.

        Returns:
            str: ``LISBOA_EMBEDDINGS_BACKEND`` when it names a backend; otherwise
            ``cloudflare`` when both Cloudflare credentials are set, else ``local``.
        """
        if cls.EMBEDDINGS_BACKEND in {"local", "cloudflare"}:
            return cls.EMBEDDINGS_BACKEND
        if cls.CLOUDFLARE_ACCOUNT_ID and cls.CLOUDFLARE_API_TOKEN:
            return "cloudflare"
        return "local"

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
    SHOW_MARKDOWN_RESPONSE_IN_TERMINAL = _env_bool(
        "SHOW_MARKDOWN_RESPONSE_IN_TERMINAL",
        default=True,
    )
    # Show detailed execution sections in the terminal summary.
    # This controls the verbose LangSmith, agent breakdown, and tool-call blocks.
    SHOW_DETAILED_EXECUTION_LOGS = _env_bool(
        "SHOW_DETAILED_EXECUTION_LOGS",
        default=True,
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
            "model": "qwen/qwen3.5-9b",
            "temperature": 0.1,
        },
        "weather": {
            "provider": "lmstudio",
            "model": "qwen/qwen3.5-9b",
            "temperature": 0,
        },
        "transport": {
            "provider": "lmstudio",
            "model": "qwen/qwen3.5-9b",
            "temperature": 0,
        },
        "researcher": {
            "provider": "lmstudio",
            "model": "qwen/qwen3.5-9b",
            "temperature": 0,
        },
        "planner": {
            "provider": "lmstudio",
            "model": "qwen/qwen3.5-9b",
            "temperature": 0.1,
        },
        "qa": {
            "provider": "lmstudio",
            "model": "qwen/qwen3.5-9b",
            "temperature": 0,
        },
    }

    # AZURE OPENAI CONFIGURATION (Cloud models)
    # .NOTE: GPT-5.x reasoning models (gpt-5.4-mini, gpt-5-mini, o1, o3, etc.) only support temperature=1
    AGENT_MODELS_AZURE = {
        "supervisor": {
            "provider": "azure",
            "model": DEFAULT_GPT_MODEL_NAME,
            "temperature": 1,
        },
        "weather": {
            "provider": "azure",
            "model": DEFAULT_GPT_MODEL_NAME,
            "temperature": 1,
        },
        "transport": {
            "provider": "azure",
            "model": DEFAULT_GPT_MODEL_NAME,
            "temperature": 1,
        },
        "researcher": {
            "provider": "azure",
            "model": DEFAULT_GPT_MODEL_NAME,
            "temperature": 1,
        },
        "planner": {
            "provider": "azure",
            "model": DEFAULT_GPT_MODEL_NAME,
            "temperature": 1,
        },
        "qa": {
            "provider": "azure",
            "model": DEFAULT_GPT_MODEL_NAME,
            "temperature": 1,
        },
    }

    # OPENAI CONFIGURATION (Direct API)
    AGENT_MODELS_OPENAI = {
        "supervisor": {
            "provider": "openai",
            "model": DEFAULT_GPT_MODEL_NAME,
            "temperature": 0.1,
        },
        "weather": {
            "provider": "openai",
            "model": DEFAULT_GPT_MODEL_NAME,
            "temperature": 0,
        },
        "transport": {
            "provider": "openai",
            "model": DEFAULT_GPT_MODEL_NAME,
            "temperature": 0,
        },
        "researcher": {
            "provider": "openai",
            "model": DEFAULT_GPT_MODEL_NAME,
            "temperature": 0,
        },
        "planner": {
            "provider": "openai",
            "model": DEFAULT_GPT_MODEL_NAME,
            "temperature": 0.1,
        },
        "qa": {
            "provider": "openai",
            "model": DEFAULT_GPT_MODEL_NAME,
            "temperature": 0,
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
            # GPT-5.x reasoning models only support temperature=1
            return {"provider": "azure", "model": cls.DEFAULT_GPT_MODEL_NAME, "temperature": 1}
        elif cls.MODEL_PROVIDER == "openai":
            return {"provider": "openai", "model": cls.DEFAULT_GPT_MODEL_NAME, "temperature": 0}
        else:  # lmstudio or default
            return {
                "provider": "lmstudio",
                "model": "qwen/qwen3.5-9b",
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
    import json

    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m📋 Configuration Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")

    # Test paths
    print("\n\033[1m📂 Data Paths:\033[0m")
    print(f"   Base Directory: {BASE_DIR}")
    print(f"   Data Collection: {Config.DATA_COLLECTION_DIR}")
    print(f"   Vector DB: {Config.VECTOR_DB_DIR}")
    print(f"   Pricing Catalog: {Config.LLM_PRICING_CATALOG_PATH}")
    print(f"   Places JSON: {Config.PATH_VISIT_LISBOA_PLACES}")
    print(f"   Events JSON: {Config.PATH_VISIT_LISBOA_EVENTS}")
    print(f"   Datasets JSON: {Config.PATH_DADOS_ABERTOS_METADATA}")

    # Check if paths exist
    print("\n\033[1m✅ Path Validation:\033[0m")
    paths_to_check = [
        ("Pricing Catalog", Config.LLM_PRICING_CATALOG_PATH),
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

    print("\n\033[1m💵 Pricing Catalog Validation:\033[0m")
    try:
        with Config.LLM_PRICING_CATALOG_PATH.open("r", encoding="utf-8") as handle:
            pricing_payload = json.load(handle)
        model_count = len(pricing_payload.get("models", {})) if isinstance(pricing_payload, dict) else 0
        snapshot_date = pricing_payload.get("pricing_snapshot_date", "n/a") if isinstance(pricing_payload, dict) else "n/a"
        print(f"   Models in catalog: \033[1;32m{model_count}\033[0m")
        print(f"   Snapshot date: {snapshot_date}")
    except (OSError, json.JSONDecodeError) as exc:
        print(f"   \033[1;31m✗ Could not load pricing catalog:\033[0m {exc}")

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
