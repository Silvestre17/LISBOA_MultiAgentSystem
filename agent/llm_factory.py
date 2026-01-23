# ==========================================================================
# Master Thesis - LLM Factory
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Implements a Factory Pattern to instantiate Large Language Models (LLMs).
#   This module provides a unified interface for creating LLM instances
#   regardless of the underlying provider (cloud or local).
# 
#   Supported Providers:
#     - LMStudio: Local server with OpenAI-compatible API (open-source models)
#     - Ollama:   Local model execution (open-source models)
#     - Groq:     High-speed inference API (qwen/llama models)
#     - Google:   Google AI Studio (Gemini models)
#     - OpenAI:   OpenAI API (GPT models)
# 
#   Design Pattern: Factory Pattern
#     - Encapsulates object creation logic
#     - Allows switching providers without changing client code
#     - Centralizes configuration and error handling
# 
#   Usage:
#     from agent.llm_factory import LLMFactory
#     llm = LLMFactory.get_llm()                   # Uses default provider from config
#     llm = LLMFactory.get_llm(provider="google")  # Override provider
# ==========================================================================

# Required libraries:
# pip install langchain-core langchain-groq langchain-google-genai langchain-openai langchain-ollama

import os
import sys
from typing import Optional, Dict, Any

from langchain_core.language_models.chat_models import BaseChatModel

# Add parent directory to path for imports
# This allows importing from the project root when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config


class LLMFactory:
    """
    Factory class for creating Language Model instances.
    
    This class implements the Factory design pattern, providing a single
    interface to create LLM instances from various providers. It handles
    API key validation, provider-specific configuration, and error messages.
    
    Supported Providers:
        - lmstudio: Local OpenAI-compatible server (recommended for development)
        - groq: High-speed cloud inference (recommended for production)
        - google: Google's Gemini models via AI Studio
        - openai: OpenAI's GPT models
        - ollama: Local model execution (for offline use)
    
    Attributes:
        None (all methods are static)
    
    Example:
        >>> from agent.llm_factory import LLMFactory
        >>> 
        >>> # Create LLM with default provider (from config)
        >>> llm = LLMFactory.get_llm()
        >>> 
        >>> # Create LLM with specific provider
        >>> llm = LLMFactory.get_llm(provider="google", temperature=0.7)
        >>> 
        >>> # Get model information
        >>> model_name = LLMFactory.get_model_info(llm)
        >>> print(f"Using model: {model_name}")
    """

    @staticmethod
    def get_llm(
        provider: str = Config.MODEL_PROVIDER,
        temperature: float = Config.TEMPERATURE,
        model: str = None  # Optional: override model from AGENT_MODELS
    ) -> BaseChatModel:
        """
        Creates and returns a configured LLM instance.
        
        This method is the main entry point for obtaining an LLM instance.
        It validates the provider, checks for required API keys, and
        configures the model with appropriate settings.
        
        Args:
            provider (str): The LLM provider to use. Options:
                - 'lmstudio': Local LM Studio server (recommended for development)
                - 'groq': Groq cloud API (fast inference, recommended for production)
                - 'google': Google AI Studio (Gemini models)
                - 'openai': OpenAI API (GPT models)
                - 'ollama': Local Ollama server
                Default: Config.MODEL_PROVIDER (from config.py)
            
            temperature (float): Controls randomness in responses.
                - 0.0: Deterministic (same input = same output)
                - 0.5: Balanced creativity
                - 1.0: Maximum creativity/randomness
                Default: Config.TEMPERATURE (from config.py)
            
            model (str, optional): Specific model name to use. If None,
                uses the default model for the provider from config.py.
                This is used by get_agent_llm() to support per-agent models.
        
        Returns:
            BaseChatModel: A configured LangChain chat model instance
                ready for use with invoke() or stream() methods.
        
        Raises:
            ValueError: If the provider is not supported or if required
                API keys are missing from environment variables.
        
        Example:
            >>> llm = LLMFactory.get_llm()
            >>> response = llm.invoke("Hello, how are you?")
            >>> print(response.content)
        """
        # Normalize provider name to lowercase
        provider = provider.lower().strip()
        
        # =====================================================================
        # PROVIDER 1: LM STUDIO (Default - Local OpenAI-compatible server)
        # =====================================================================
        # LM Studio provides a local server with OpenAI-compatible API.
        # Recommended for: Development, offline use, privacy-sensitive data
        # Setup: Download LM Studio, load a model, start local server
        # Default URL: http://localhost:1234/v1
        if provider == "lmstudio":
            # Import OpenAI integration (LM Studio uses OpenAI-compatible API)
            from langchain_openai import ChatOpenAI
            
            # Use provided model or fall back to default
            model_name = model if model else Config.LMSTUDIO_MODEL_NAME
            
            return ChatOpenAI(
                model=model_name,  # e.g., "qwen/qwen3-4b-2507"
                temperature=temperature,
                base_url=Config.LMSTUDIO_BASE_URL,  # e.g., "http://localhost:1234/v1"
                api_key="lm-studio",  # LM Studio ignores API key, any string works
                # Add penalties to prevent repetition (critical for small models)
                frequency_penalty=0.5,
                presence_penalty=0.3,
            )
        
        # =====================================================================
        # PROVIDER 2: GROQ (High-speed inference)
        # =====================================================================
        # Groq provides extremely fast inference for open-source models.
        # Recommended for: Production use, real-time chat applications
        # Models available: Qwen, Llama, Mixtral
        # Free tier: 14,400 requests/day
        elif provider == "groq":
            # Validate API key exists
            if not Config.GROQ_API_KEY:
                raise ValueError(
                    "\033[1;31m❌ GROQ_API_KEY not found in environment.\033[0m\n"
                    "   To fix: Add GROQ_API_KEY to your .env file\n"
                    "   Get your free key at: https://console.groq.com"
                )
            
            # Import Groq-specific LangChain integration
            from langchain_groq import ChatGroq
            
            model_name = model if model else Config.GROQ_MODEL_NAME
            
            return ChatGroq(
                model_name=model_name,  # e.g., "llama-3.3-70b-versatile"
                temperature=temperature,
                api_key=Config.GROQ_API_KEY
            )
        
        # =====================================================================
        # PROVIDER 3: GOOGLE (Gemini models)
        # =====================================================================
        # Google AI Studio provides access to Gemini models.
        # Recommended for: Tasks requiring multimodal capabilities
        # Models available: gemini-2.0-flash-exp, gemini-1.5-pro
        # Free tier: 60 requests/minute
        elif provider == "google":
            # Validate API key exists
            if not Config.GOOGLE_API_KEY:
                raise ValueError(
                    "\033[1;31m❌ GOOGLE_API_KEY not found in environment.\033[0m\n"
                    "   To fix: Add GOOGLE_API_KEY to your .env file\n"
                    "   Get your key at: https://aistudio.google.com/apikey"
                )
            
            # Import Google-specific LangChain integration
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            model_name = model if model else Config.GOOGLE_MODEL_NAME
            
            return ChatGoogleGenerativeAI(
                model=model_name,  # e.g., "gemini-2.0-flash-exp"
                temperature=temperature,
                google_api_key=Config.GOOGLE_API_KEY,
                # Gemini doesn't support system messages natively
                # This flag converts them to human messages
                convert_system_message_to_human=True
            )
        
        # =====================================================================
        # PROVIDER 4: OPENAI (GPT models)
        # =====================================================================
        # OpenAI API provides access to GPT models.
        # Recommended for: Tasks requiring highest quality outputs
        # Models available: gpt-4o, gpt-4o-mini, gpt-4-turbo
        # Pricing: Pay-per-use (no free tier for GPT-4)
        elif provider == "openai":
            # Validate API key exists
            if not Config.OPENAI_API_KEY:
                raise ValueError(
                    "\033[1;31m❌ OPENAI_API_KEY not found in environment.\033[0m\n"
                    "   To fix: Add OPENAI_API_KEY to your .env file\n"
                    "   Get your key at: https://platform.openai.com/api-keys"
                )
            
            # Import OpenAI-specific LangChain integration
            from langchain_openai import ChatOpenAI
            
            model_name = model if model else Config.OPENAI_MODEL_NAME
            
            return ChatOpenAI(
                model=model_name,  # e.g., "gpt-4o-mini"
                temperature=temperature,
                api_key=Config.OPENAI_API_KEY
            )
        
        # =====================================================================
        # PROVIDER 5: OLLAMA (Local model execution)
        # =====================================================================
        # Ollama runs models locally on your machine.
        # Recommended for: Offline use, development, testing
        # Setup: Install Ollama, pull model (ollama pull qwen2.5:7b), run server
        # Server command: ollama serve
        elif provider == "ollama":
            # Import Ollama-specific LangChain integration
            from langchain_ollama import ChatOllama
            
            model_name = model if model else Config.OLLAMA_MODEL_NAME
            
            return ChatOllama(
                model=model_name,  # e.g., "qwen2.5:7b"
                temperature=temperature,
                keep_alive="5m"  # Keep model loaded for 5 minutes after last request
            )
        
        # =====================================================================
        # UNSUPPORTED PROVIDER
        # =====================================================================
        else:
            raise ValueError(
                f"\033[1;31m❌ Unsupported provider: '{provider}'\033[0m\n"
                f"   Supported providers: lmstudio, groq, google, openai, ollama\n"
                f"   Default provider: {Config.MODEL_PROVIDER}"
            )
    
    @staticmethod
    def get_model_info(llm: BaseChatModel) -> Dict[str, Any]:
        """
        Extracts comprehensive information from an LLM instance.
        
        Returns a dictionary with all relevant configuration details including
        model name, temperature, penalty parameters, and provider-specific settings.
        Handles various providers (OpenAI, Groq, Google, Ollama) dynamically.
        
        Args:
            llm (BaseChatModel): The LLM instance to inspect.
        
        Returns:
            Dict[str, Any]: Comprehensive configuration dictionary.
            
        Example:
            >>> llm = LLMFactory.get_llm()
            >>> info = LLMFactory.get_model_info(llm)
            >>> print(info['frequency_penalty'])
            0.5
        """
        info = {
            "type": llm.__class__.__name__,
            "model": "Unknown"
        }
        
        # 1. Standard Attributes to inspect
        # Note: Different versions of LangChain might store these differently
        standard_attrs = [
            "model_name", "model", 
            "temperature", 
            "max_tokens", 
            "top_p", "top_k", 
            "frequency_penalty", "presence_penalty", 
            "n", 
            "streaming", 
            "max_retries", 
            "request_timeout", 
            "base_url", "openai_api_base", "api_base", # various URL attributes
            "timeout"
        ]
        
        for attr in standard_attrs:
            if hasattr(llm, attr):
                val = getattr(llm, attr)
                if val is not None:
                    # Normalize model name key
                    if attr in ["model_name", "model"]:
                        info["model"] = val
                    # Normalize base_url key
                    elif attr in ["base_url", "openai_api_base", "api_base"]:
                        info["base_url"] = val
                    else:
                        info[attr] = val

        # 2. Inspect model_kwargs (Common bucket for extra params)
        if hasattr(llm, "model_kwargs") and isinstance(llm.model_kwargs, dict):
            # We add these but don't overwrite existing high-level keys if they exist
            # This captures provider-specific params passed during init
            for k, v in llm.model_kwargs.items():
                if k not in info and v is not None:
                    info[k] = v

        # 3. Provider-Specific Extraction
        
        # Google/Gemini specific
        if "Google" in info["type"]:
            # Check for generation_config
            gen_config = getattr(llm, "generation_config", None)
            if gen_config:
                if isinstance(gen_config, dict):
                    info.update({k: v for k, v in gen_config.items() if v is not None})
                elif hasattr(gen_config, "__dict__"):
                     try:
                        info.update({k: v for k, v in vars(gen_config).items() if v is not None})
                     except:
                        pass
        
        # Ollama specific
        if "Ollama" in info["type"]:
            if hasattr(llm, "keep_alive"):
                info["keep_alive"] = llm.keep_alive
            if hasattr(llm, "num_ctx"):
                 info["num_ctx"] = llm.num_ctx
        
        return info
    
    @staticmethod
    def get_agent_llm(agent_name: str) -> BaseChatModel:
        """
        Creates an LLM instance configured for a specific agent.
        
        Uses AGENT_MODELS from config.py to get per-agent model configuration.
        This allows different agents to use different models/providers.
        
        Args:
            agent_name (str): Name of the agent (e.g., 'supervisor', 'weather',
                            'transport', 'researcher', 'planner').
        
        Returns:
            BaseChatModel: Configured LLM for the specified agent.
        
        Example:
            >>> llm = LLMFactory.get_agent_llm("supervisor")
            >>> # Returns LLM configured with supervisor's provider/model
        """
        # Get agent-specific config or fallback to default
        agent_config = Config.AGENT_MODELS.get(agent_name, Config.DEFAULT_AGENT_MODEL)
        
        provider = agent_config.get("provider", Config.MODEL_PROVIDER)
        model = agent_config.get("model", None)  # Get specific model name
        temperature = agent_config.get("temperature", Config.TEMPERATURE)
        
        # Create LLM with the agent's configured provider/temperature/model
        return LLMFactory.get_llm(
            provider=provider, 
            temperature=temperature,
            model=model
        )


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    """
    Test script for the LLM Factory.
    
    This script tests the LLM Factory by:
        1. Creating an LLM instance with the default provider
        2. Displaying the model information
        3. Sending a simple test prompt
        4. Displaying the response
    
    Expected Output (with valid API key):
        - LLM successfully initialized
        - Model name displayed
        - Response to test prompt shown
    
    Expected Output (without valid API key):
        - Error message with troubleshooting tips
    """
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 LLM Factory Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    try:
        # Get the current provider from configuration
        provider = Config.MODEL_PROVIDER
        print(f"\n\033[1m🔄 Initializing LLM...\033[0m")
        print(f"   Provider: {provider}")
        
        # Create LLM instance using factory
        llm = LLMFactory.get_llm()
        model_info = LLMFactory.get_model_info(llm)
        
        print(f"\n\033[1;32m✅ LLM Ready:\033[0m {model_info['model']}")
        print(f"   Details: {model_info}")
        
        # Test with a simple prompt
        print(f"\n\033[1m🧪 Testing with prompt...\033[0m")
        prompt = "What is the capital of Portugal? Answer in one sentence."
        print(f"   Prompt: '{prompt}'")
        
        response = llm.invoke(prompt)
        
        print(f"\n\033[1m🤖 Response:\033[0m")
        print(f"   {response.content}")
        
        print(f"\n\033[1;32m✅ Test passed!\033[0m")
        
    except Exception as e:
        # Display error with troubleshooting tips
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
        print("\n\033[1m💡 Troubleshooting Tips:\033[0m")
        print("   1. Verify API keys are correctly set in .env file")
        print("   2. For LM Studio: Ensure server is running on port 1234")
        print("   3. For Ollama: Run 'ollama serve' before testing")
        print("   4. Check internet connection for cloud providers")