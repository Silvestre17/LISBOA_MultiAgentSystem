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
#     - OpenAI:   OpenAI API (GPT models)
#     - Azure:    Azure OpenAI Service (enterprise cloud)
#
#   Design Pattern: Factory Pattern
#     - Encapsulates object creation logic
#     - Allows switching providers without changing client code
#     - Centralizes configuration and error handling
#
#   Usage:
#     from agent.llm_factory import LLMFactory
#     llm = LLMFactory.get_llm()                   # Uses default provider from config
#     llm = LLMFactory.get_llm(provider="openai")  # Override provider
# ==========================================================================

# Required libraries:
# pip install langchain-core langchain-openai

from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agent.utils.langsmith_tracing import LANGSMITH_STATUS

try:
    from config import Config
except ModuleNotFoundError:
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import Config


class LLMFactory:
    """
    Factory class for creating Language Model instances.

    This class implements the Factory design pattern, providing a single
    interface to create LLM instances from various providers. It handles
    API key validation, provider-specific configuration, and error messages.

    Supported Providers:
        - lmstudio: Local OpenAI-compatible server (recommended for development)
        - openai: OpenAI API (GPT models)
        - azure: Azure OpenAI Service (enterprise cloud)

    Attributes:
        None (all methods are static)

    Example:
        >>> from agent.llm_factory import LLMFactory
        >>>
        >>> # Create LLM with default provider (from config)
        >>> llm = LLMFactory.get_llm()
        >>>
        >>> # Create LLM with specific provider
        >>> llm = LLMFactory.get_llm(provider="openai", temperature=0.7)
        >>>
        >>> # Get model information
        >>> model_name = LLMFactory.get_model_info(llm)
        >>> print(f"Using model: {model_name}")
    """

    @staticmethod
    def _is_reasoning_model(model_name: str) -> bool:
        """
        Checks if a model is a reasoning model that only supports temperature=1.

        The current heuristic treats o-series models and most `gpt-5` family
        deployments as reasoning-style models. Explicit chat variants such as
        `gpt-5-chat` are excluded from that rule.
        
        Args:
            model_name (str): The model name to check.
            
        Returns:
            bool: True if this is a reasoning model, False otherwise.
        """
        model_lower = model_name.lower()
        
        # Check for o-series models (o1, o3, o4)
        is_o_series = any(x in model_lower for x in ["o1", "o3", "o4", "o-"])
        
        # Check for gpt-5 (but NOT gpt-5-chat which supports temperature)
        is_gpt5_reasoning = (
            "gpt-5" in model_lower and "gpt-5-chat" not in model_lower
        )
        
        return is_o_series or is_gpt5_reasoning

    @staticmethod
    def get_llm(
        provider: str = Config.MODEL_PROVIDER,
        temperature: float = Config.TEMPERATURE,
        model: Optional[str] = None,  # Optional: override model from AGENT_MODELS
    ) -> BaseChatModel:
        """
        Creates and returns a configured LLM instance.

        This method is the main entry point for obtaining an LLM instance.
        It validates the provider, checks for required API keys, and
        configures the model with appropriate settings.

        Args:
            provider (str): The LLM provider to use. Options:
                - 'lmstudio': Local LM Studio server (recommended for development)
                - 'openai': OpenAI API (GPT models)
                - 'azure': Azure OpenAI Service (enterprise cloud)
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
            model_name = model or Config.LMSTUDIO_MODEL_NAME or Config.get_default_agent_model().get("model")

            return ChatOpenAI(
                model=model_name,  # e.g., "qwen/qwen3-4b-2507"
                temperature=temperature,
                base_url=Config.LMSTUDIO_BASE_URL,  # e.g., "http://localhost:1234/v1"
                api_key="lm-studio",  # LM Studio ignores API key, any string works
                # Add penalties to prevent repetition (critical for small models)
                frequency_penalty=0.5,
                presence_penalty=0.3,
                streaming=True,  # Enable streaming for faster first-token latency
                stream_usage=True,  # Report token usage for LangSmith cost tracking
            )

        # =====================================================================
        # PROVIDER 2: OPENAI (GPT models)
        # =====================================================================
        # OpenAI API provides access to GPT models.
        # Recommended for: Tasks requiring highest quality outputs
        # Models available: gpt-5.2, gpt-5.1, gpt-5, gpt-5-mini, gpt-5-nano
        # Pricing: Pay-per-use
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

            model_name = model or Config.OPENAI_MODEL_NAME or Config.get_default_agent_model().get("model")

            # Check if it's a reasoning-style deployment using helper function
            if LLMFactory._is_reasoning_model(model_name):
                # Reasoning models only support temperature=1, omit the parameter
                return ChatOpenAI(
                    model=model_name,
                    api_key=Config.OPENAI_API_KEY,
                    streaming=True,  # Enable streaming for lower latency
                    stream_usage=True,  # Report token usage for LangSmith cost tracking
                )
            else:
                return ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    api_key=Config.OPENAI_API_KEY,
                    streaming=True,  # Enable streaming for lower latency
                    stream_usage=True,  # Report token usage for LangSmith cost tracking
                )

        # =====================================================================
        # PROVIDER 3: AZURE OPENAI (v1 API - uses /openai/v1/ endpoint)
        # =====================================================================
        # Azure OpenAI via new v1 API (August 2025+)
        # Uses ChatOpenAI with base_url instead of AzureChatOpenAI
        # Benefits: No api_version needed, faster updates, OpenAI-compatible
        # Required env vars: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
        # .IMPORTANT: o-series models (gpt-5-nano, o1, o3) only support temperature=1
        elif provider == "azure":
            if not Config.AZURE_OPENAI_API_KEY:
                raise ValueError("AZURE_OPENAI_API_KEY not found in .env file")
            if not Config.AZURE_OPENAI_ENDPOINT:
                raise ValueError("AZURE_OPENAI_ENDPOINT not found in .env file")

            from langchain_openai import ChatOpenAI

            # Determine the model being used
            model_name = (
                model
                or Config.AZURE_OPENAI_DEPLOYMENT_NAME
                or Config.get_default_agent_model().get("model")
            )

            # Build the v1 API base URL
            # Format: https://YOUR-RESOURCE.openai.azure.com/openai/v1/
            endpoint = Config.AZURE_OPENAI_ENDPOINT.rstrip("/")
            base_url = f"{endpoint}/openai/v1/"

            # Check if it's a reasoning-style deployment using helper function
            if LLMFactory._is_reasoning_model(model_name):
                # Reasoning-style models are configured without temperature here
                # Use minimal reasoning effort for lower latency
                # max_completion_tokens for optimal output capacity
                return ChatOpenAI(
                    model=model_name,  # Deployment name
                    api_key=Config.AZURE_OPENAI_API_KEY,
                    base_url=base_url,  # v1 API endpoint
                    max_completion_tokens=16384,  # Optimal token limit
                    streaming=True,  # Enable streaming by default
                    stream_usage=True,  # Report token usage for LangSmith cost tracking
                    timeout=60,  # 60 second timeout for faster failure detection
                    max_retries=2,  # Reduced retries for faster failure
                    reasoning_effort="minimal",  # Minimal for lower latency
                )
            else:
                # Standard models (including gpt-5-chat) support temperature
                return ChatOpenAI(
                    model=model_name,  # Deployment name
                    api_key=Config.AZURE_OPENAI_API_KEY,
                    base_url=base_url,  # v1 API endpoint
                    temperature=temperature,
                    max_completion_tokens=16384,  # Optimal token limit
                    streaming=True,  # Enable streaming by default
                    stream_usage=True,  # Report token usage for LangSmith cost tracking
                    timeout=60,  # 60 second timeout for faster failure detection
                    max_retries=2,  # Reduced retries for faster failure
                )

        # =====================================================================
        # UNSUPPORTED PROVIDER
        # =====================================================================
        else:
            raise ValueError(
                f"\033[1;31m❌ Unsupported provider: '{provider}'\033[0m\n"
                f"   Supported providers: lmstudio, openai, azure\n"
                f"   Default provider: {Config.MODEL_PROVIDER}"
            )

    @staticmethod
    def get_model_info(llm: BaseChatModel) -> Dict[str, Any]:
        """
        Extracts comprehensive information from an LLM instance.

        Returns a dictionary with all relevant configuration details including
        model name, temperature, penalty parameters, and provider-specific settings.
        Handles various providers (OpenAI, Azure, LM Studio) dynamically.

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
        info = {"type": llm.__class__.__name__, "model": "Unknown"}

        # 1. Standard Attributes to inspect
        # Note: Different versions of LangChain might store these differently
        standard_attrs = [
            "model_name",
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "n",
            "streaming",
            "max_retries",
            "request_timeout",
            "base_url",
            "openai_api_base",
            "api_base",  # various URL attributes
            "timeout",
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

        return info

    @staticmethod
    def extract_usage_metadata(response: Any) -> Dict[str, Any]:
        """
        Extracts normalized token usage metadata from a model response.

        Supports raw LangChain AIMessage responses, structured-output wrappers
        with ``include_raw=True``, and provider-specific token usage fields such
        as ``prompt_tokens`` / ``completion_tokens``.

        Args:
            response: Raw model response or structured-output wrapper.

        Returns:
            Dict[str, Any]: Normalized usage payload with ``input_tokens``,
            ``output_tokens``, ``total_tokens``, and ``usage_available``.
        """
        raw_response = response.get("raw") if isinstance(response, dict) else response

        usage_candidates = []
        if hasattr(raw_response, "usage_metadata") and raw_response.usage_metadata:
            usage_candidates.append(raw_response.usage_metadata)

        response_metadata = getattr(raw_response, "response_metadata", None)
        if isinstance(response_metadata, dict):
            for key in ("token_usage", "usage", "usage_metadata"):
                candidate = response_metadata.get(key)
                if isinstance(candidate, dict):
                    usage_candidates.append(candidate)

        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        usage_available = False

        def _coerce_int(value: Any) -> int:
            """Coerces token counts to integers safely."""
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        for candidate in usage_candidates:
            if not isinstance(candidate, dict):
                continue

            usage_available = True
            input_tokens = max(
                input_tokens,
                _coerce_int(
                    candidate.get("input_tokens", candidate.get("prompt_tokens", candidate.get("input_token_count")))
                ),
            )
            output_tokens = max(
                output_tokens,
                _coerce_int(
                    candidate.get(
                        "output_tokens",
                        candidate.get("completion_tokens", candidate.get("output_token_count")),
                    )
                ),
            )
            total_tokens = max(
                total_tokens,
                _coerce_int(candidate.get("total_tokens", candidate.get("total_token_count"))),
            )

        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        if input_tokens or output_tokens or total_tokens:
            usage_available = True

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "usage_available": usage_available,
        }

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
        agent_config = Config.AGENT_MODELS().get(
            agent_name, Config.DEFAULT_AGENT_MODEL()
        )

        provider = agent_config.get("provider", Config.MODEL_PROVIDER)
        model = agent_config.get("model", None)  # Get specific model name
        temperature = agent_config.get("temperature", Config.TEMPERATURE)

        # Create LLM with the agent's configured provider/temperature/model
        return LLMFactory.get_llm(
            provider=provider, temperature=temperature, model=model
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
        print("\n\033[1m🔄 Initializing LLM...\033[0m")
        print(f"   Provider: {provider}")

        # Create LLM instance using factory
        default_model = Config.get_default_agent_model().get("model")
        llm = LLMFactory.get_llm(provider=provider, model=default_model)
        model_info = LLMFactory.get_model_info(llm)

        print(f"\n\033[1;32m✅ LLM Ready:\033[0m {model_info['model']}")
        print(f"   Details: {model_info}")

        # Test with a simple prompt
        print("\n\033[1m🧪 Testing with prompt...\033[0m")
        prompt = "What is the capital of Portugal? Answer in one sentence."
        print(f"   Prompt: '{prompt}'")

        response = llm.invoke(prompt)

        print("\n\033[1m🤖 Response:\033[0m")
        print(f"   {response.content}")

        print("\n\033[1;32m✅ Test passed!\033[0m")

    except Exception as e:
        # Display error with troubleshooting tips
        print(f"\n\033[1;31m❌ Error:\033[0m {e}")
        print("\n\033[1m💡 Troubleshooting Tips:\033[0m")
        print("   1. Verify API keys are correctly set in .env file")
        print("   2. For LM Studio: Ensure server is running on port 1234")
        print("   3. For Azure: Check your Azure OpenAI endpoint configuration")
        print("   4. Check internet connection for cloud providers")
