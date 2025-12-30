# ==========================================================================
# Master Thesis - LLM Factory
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Implements a Factory Pattern to instantiate Large Language Models (LLMs).
#   This module provides a unified interface for creating LLM instances
#   regardless of the underlying provider (cloud or local).
# 
#   Supported Providers:
#     - LMStudio: Local server with OpenAI-compatible API (Default: qwen/qwen3-4b-2507)
#     - Groq:     High-speed inference API (qwen/llama models)
#     - Google:   Google AI Studio (Gemini models)
#     - OpenAI:   OpenAI API (GPT models)
#     - Ollama:   Local model execution
# 
#   Design Pattern: Factory Pattern
#     - Encapsulates object creation logic
#     - Allows switching providers without changing client code
#     - Centralizes configuration and error handling
# 
#   Usage:
#     from agent.llm_factory import LLMFactory
#     llm = LLMFactory.get_llm()  # Uses default provider from config
#     llm = LLMFactory.get_llm(provider="google")  # Override provider
# ==========================================================================

# Required libraries:
# pip install langchain-core langchain-groq langchain-google-genai langchain-openai langchain-ollama

import os
import sys
from typing import Optional

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
        temperature: float = Config.TEMPERATURE
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
            
            return ChatOpenAI(
                model=Config.LMSTUDIO_MODEL_NAME,  # e.g., "qwen/qwen3-4b-2507"
                temperature=temperature,
                base_url=Config.LMSTUDIO_BASE_URL,  # e.g., "http://localhost:1234/v1"
                api_key="lm-studio"  # LM Studio ignores API key, any string works
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
            
            return ChatGroq(
                model_name=Config.GROQ_MODEL_NAME,  # e.g., "llama-3.3-70b-versatile"
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
            
            return ChatGoogleGenerativeAI(
                model=Config.GOOGLE_MODEL_NAME,  # e.g., "gemini-2.0-flash-exp"
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
            
            return ChatOpenAI(
                model=Config.OPENAI_MODEL_NAME,  # e.g., "gpt-4o-mini"
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
            
            return ChatOllama(
                model=Config.OLLAMA_MODEL_NAME,  # e.g., "qwen2.5:7b"
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
    def get_model_info(llm: BaseChatModel) -> str:
        """
        Extracts and returns the model name from an LLM instance.
        
        Different LLM providers store the model name in different attributes.
        This method provides a unified way to retrieve the model name
        regardless of the provider.
        
        Args:
            llm (BaseChatModel): The LLM instance to inspect.
        
        Returns:
            str: The model name (e.g., "qwen/qwen3-4b-2507", "gpt-4o-mini").
                Returns "Unknown" if the model name cannot be determined.
        
        Example:
            >>> llm = LLMFactory.get_llm()
            >>> print(LLMFactory.get_model_info(llm))
            'qwen/qwen3-4b-2507'
        """
        # Try 'model_name' first (used by Groq, Ollama)
        # Then try 'model' (used by OpenAI, Google)
        # Default to "Unknown" if neither exists
        model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'Unknown'))
        return model_name


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
        model_name = LLMFactory.get_model_info(llm)
        
        print(f"\n\033[1;32m✅ LLM Ready:\033[0m {model_name}")
        
        # Test with a simple prompt
        print(f"\n\033[1m🧪 Testing with prompt...\033[0m")
        print("   Prompt: 'What is the capital of Portugal? Answer in one sentence.'")
        
        response = llm.invoke("What is the capital of Portugal? Answer in one sentence.")
        
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