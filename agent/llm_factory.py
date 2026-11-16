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
#     - Visitor connection: any OpenAI-compatible endpoint supplied in the
#       Streamlit sidebar for one browser session (LLMConnection)
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

import hashlib
import ipaddress
import re
import socket
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import httpcore
import httpx
from langchain_core.language_models.chat_models import BaseChatModel

try:
    from config import Config
except ModuleNotFoundError:
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import Config


# Hosts treated as a local runtime: agents run sequentially and system
# messages are merged, as for the LM Studio provider.
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

# Placeholder sent when a local server needs no key. An explicit value stops
# the OpenAI client from falling back to the server's OPENAI_API_KEY.
_NO_API_KEY = "not-needed"


def _azure_v1_base_url(endpoint: str) -> str:
    """Return the Azure OpenAI v1 base URL for a resource endpoint.

    Args:
        endpoint: Resource endpoint, with or without ``/openai`` or ``/openai/v1``.

    Returns:
        str: Base URL ending in ``/openai/v1/``.
    """
    endpoint = endpoint.rstrip("/")
    endpoint_lower = endpoint.lower()
    if endpoint_lower.endswith("/openai/v1"):
        return f"{endpoint}/"
    if endpoint_lower.endswith("/openai"):
        return f"{endpoint}/v1/"
    return f"{endpoint}/openai/v1/"


def _resolve_public_addresses(host: str) -> List[str]:
    """Resolve a host and require every address to be public.

    IP literals resolve to themselves, so loopback, private, link-local,
    reserved and multicast literals are refused as well as names that point
    to them.

    Args:
        host: Host name or IP literal from the visitor's base URL.

    Returns:
        List[str]: The resolved addresses, in resolver order.

    Raises:
        ValueError: ``private_host`` for a non-public address, or
            ``connection`` when the host does not resolve.
    """
    try:
        address_infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError):
        raise ValueError("connection") from None
    addresses: List[str] = []
    for address_info in address_infos:
        literal = address_info[4][0].split("%", 1)[0]
        address = ipaddress.ip_address(literal)
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if not address.is_global or address.is_multicast or address.is_reserved:
            raise ValueError("private_host")
        if literal not in addresses:
            addresses.append(literal)
    return addresses


class _PublicAddressBackend(httpcore.NetworkBackend):
    """Network backend that opens connections to public addresses only.

    The host is resolved again for every new connection and the socket goes
    to one of the addresses just checked, so a DNS answer that changes after
    the settings form validated it (DNS rebinding) cannot reach a private
    address. TLS still verifies the certificate for the original host name.
    """

    def __init__(self) -> None:
        self._backend = httpcore.SyncBackend()

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        """Connect to the first reachable public address of ``host``."""
        try:
            addresses = _resolve_public_addresses(host)
        except ValueError as exc:
            raise httpcore.ConnectError(str(exc)) from None
        last_error: Exception = httpcore.ConnectError("connection")
        for address in addresses:
            try:
                return self._backend.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_error = exc
        raise last_error

    def connect_unix_socket(self, path, timeout=None, socket_options=None):
        """Refuse Unix sockets: visitor endpoints are network hosts."""
        raise httpcore.ConnectError("private_host")

    def sleep(self, seconds: float) -> None:
        """Delegate retry back-off sleeps to the standard backend."""
        self._backend.sleep(seconds)


class _PublicAddressTransport(httpx.HTTPTransport):
    """httpx transport whose connections go through ``_PublicAddressBackend``.

    httpx does not expose the network backend, so the connection pool built
    by ``HTTPTransport`` is replaced; any other layout fails closed.
    """

    def __init__(self) -> None:
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
        super().__init__(trust_env=False, limits=limits)
        if not isinstance(self._pool, httpcore.ConnectionPool):
            raise RuntimeError("Unexpected httpx transport layout; visitor connections are disabled.")
        self._pool.close()
        self._pool = httpcore.ConnectionPool(
            ssl_context=httpx.create_ssl_context(trust_env=False),
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            network_backend=_PublicAddressBackend(),
        )


class _RefusingAsyncTransport(httpx.AsyncBaseTransport):
    """Async transport that refuses every request.

    The agents call models synchronously. Without this, an async call would
    use the OpenAI default client, which follows redirects and skips the
    address checks.
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Refuse the request."""
        raise httpx.ConnectError("Asynchronous visitor model calls are disabled.", request=request)


_VISITOR_HTTP_CLIENTS: Optional[tuple[httpx.Client, httpx.AsyncClient]] = None
_VISITOR_HTTP_CLIENTS_LOCK = threading.Lock()


def _visitor_http_clients() -> tuple[httpx.Client, httpx.AsyncClient]:
    """Return the shared HTTP clients for visitor endpoints on hosted deployments.

    They never follow redirects and ignore proxy environment variables, so a
    visitor endpoint cannot bounce the server to an internal address. The
    keys travel in per-request headers, so sharing the pool is safe.
    """
    global _VISITOR_HTTP_CLIENTS
    with _VISITOR_HTTP_CLIENTS_LOCK:
        if _VISITOR_HTTP_CLIENTS is None:
            _VISITOR_HTTP_CLIENTS = (
                httpx.Client(transport=_PublicAddressTransport(), follow_redirects=False, trust_env=False),
                httpx.AsyncClient(transport=_RefusingAsyncTransport(), follow_redirects=False, trust_env=False),
            )
        return _VISITOR_HTTP_CLIENTS


@dataclass(frozen=True)
class LLMConnection:
    """OpenAI-compatible endpoint that a visitor supplies for one session.

    The key lives only in this object, which the app keeps in the visitor's
    ``st.session_state``. It is passed straight to the chat-model client and is
    never written to the environment, ``Config``, files or logs. ``repr`` hides it.

    Attributes:
        base_url: Endpoint base URL, for example ``https://api.openai.com/v1``.
        model: Model or deployment name sent with every request.
        api_key: Visitor's key; empty for local servers that need none.
    """

    base_url: str
    model: str
    api_key: str = field(default="", repr=False)

    @classmethod
    def create(cls, base_url: str, model: str, api_key: str = "") -> "LLMConnection":
        """Validate and normalize visitor input into a connection.

        Azure resource endpoints are completed to their ``/openai/v1/`` path.
        Unless ``Config.ALLOW_PRIVATE_LLM_HOSTS`` is set, the host must resolve
        only to public addresses.

        Args:
            base_url: Base URL typed by the visitor.
            model: Model name typed by the visitor.
            api_key: Key typed by the visitor.

        Returns:
            LLMConnection: Normalized connection.

        Raises:
            ValueError: ``invalid_base_url``, ``missing_model``, ``private_host``
                or ``connection``; the message is a reason code that never
                contains the input values.
        """
        base_url = (base_url or "").strip()
        model = (model or "").strip()
        try:
            parts = urlsplit(base_url)
        except ValueError:
            raise ValueError("invalid_base_url") from None
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("invalid_base_url")
        if not model:
            raise ValueError("missing_model")
        if not Config.ALLOW_PRIVATE_LLM_HOSTS:
            _resolve_public_addresses(parts.hostname)
        if parts.hostname.endswith(".azure.com") and "/v1" not in parts.path:
            base_url = _azure_v1_base_url(base_url)
        return cls(base_url=base_url, model=model, api_key=(api_key or "").strip())

    @property
    def provider(self) -> str:
        """Provider label for agents and usage logs: ``local`` or ``compatible``."""
        host = (urlsplit(self.base_url).hostname or "").lower()
        is_local = host in _LOOPBACK_HOSTS or host.endswith(".localhost")
        return "local" if is_local else "compatible"

    @property
    def fingerprint(self) -> str:
        """Non-reversible identifier of the endpoint, model and key."""
        key_hash = hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()
        material = "\n".join((self.base_url, self.model, key_hash))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


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

        # Check for o-series models (o1, o3, o4) without misclassifying
        # chat models such as gpt-4o-mini, where the o is part of 4o.
        is_o_series = bool(re.search(r"(^|[^a-z0-9])o(?:1|3|4)(?:$|[^a-z0-9])", model_lower))

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
                model=model_name,  # e.g., "qwen/qwen3.5-9b"
                temperature=temperature,
                base_url=Config.LMSTUDIO_BASE_URL,  # e.g., "http://localhost:1234/v1"
                api_key="lm-studio",  # LM Studio ignores API key, any string works
                max_tokens=2048,
                timeout=180,
                max_retries=1,
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
        # Models available: gpt-5.2, gpt-5.1, gpt-5, gpt-5-mini
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
                    streaming=True,     # Enable streaming for lower latency
                    stream_usage=True,  # Report token usage for LangSmith cost tracking
                    timeout=180,
                    max_retries=2,
                )
            else:
                return ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    api_key=Config.OPENAI_API_KEY,
                    streaming=True,     # Enable streaming for lower latency
                    stream_usage=True,  # Report token usage for LangSmith cost tracking
                    timeout=180,
                    max_retries=2,
                )

        # =====================================================================
        # PROVIDER 3: AZURE OPENAI (v1 API - uses /openai/v1/ endpoint)
        # =====================================================================
        # Azure OpenAI via new v1 API (August 2025+)
        # Uses ChatOpenAI with base_url instead of AzureChatOpenAI
        # Benefits: No api_version needed, faster updates, OpenAI-compatible
        # Required env vars: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
        # .IMPORTANT: o-series models (gpt-5-mini, o1, o3) only support temperature=1
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
            base_url = _azure_v1_base_url(Config.AZURE_OPENAI_ENDPOINT)

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
                    timeout=180,  # Long enough for planning and QA synthesis
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
                    timeout=180,  # Long enough for planning and QA synthesis
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
    def server_provider_configured() -> bool:
        """Return whether the server provider in ``config.py`` has its credentials.

        LM Studio needs no key, so it counts as configured; its server is only
        checked by the first request.

        Returns:
            bool: True when the server default provider can be built.
        """
        provider = Config.MODEL_PROVIDER.lower().strip()
        if provider == "azure":
            return bool(Config.AZURE_OPENAI_API_KEY and Config.AZURE_OPENAI_ENDPOINT)
        if provider == "openai":
            return bool(Config.OPENAI_API_KEY)
        return provider == "lmstudio"

    @staticmethod
    def get_connection_llm(
        connection: LLMConnection,
        temperature: float = Config.TEMPERATURE,
        timeout: float = 180,
        max_retries: int = 2,
    ) -> BaseChatModel:
        """Create a chat model for a visitor's OpenAI-compatible endpoint.

        The key is always passed explicitly, so the OpenAI client never falls
        back to server environment variables such as ``OPENAI_API_KEY``.
        Unless ``Config.ALLOW_PRIVATE_LLM_HOSTS`` is set, requests go through
        clients that re-check the host at every connection and never follow
        redirects.

        Args:
            connection: Visitor endpoint, model and key for this session.
            temperature: Sampling temperature; omitted for reasoning models.
            timeout: Request timeout in seconds.
            max_retries: Client retries for transient failures.

        Returns:
            BaseChatModel: Configured chat model.
        """
        from langchain_openai import ChatOpenAI

        params: Dict[str, Any] = {
            "model": connection.model,
            "base_url": connection.base_url,
            "api_key": connection.api_key or _NO_API_KEY,
            "streaming": True,
            "stream_usage": True,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        # Reasoning effort stays at the provider default: accepted values differ
        # between models and providers, and a rejected value would fail every call.
        if not LLMFactory._is_reasoning_model(connection.model):
            params["temperature"] = temperature
        if not Config.ALLOW_PRIVATE_LLM_HOSTS:
            params["http_client"], params["http_async_client"] = _visitor_http_clients()
        return ChatOpenAI(**params)

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
    def get_agent_llm(
        agent_name: str,
        connection: Optional[LLMConnection] = None,
    ) -> BaseChatModel:
        """
        Creates an LLM instance configured for a specific agent.

        Uses AGENT_MODELS from config.py to get per-agent model configuration.
        This allows different agents to use different models/providers.

        Args:
            agent_name (str): Name of the agent (e.g., 'supervisor', 'weather',
                            'transport', 'researcher', 'planner').
            connection: Visitor endpoint for this session. When set, every
                agent uses its model, and the server provider is not touched.

        Returns:
            BaseChatModel: Configured LLM for the specified agent.

        Example:
            >>> llm = LLMFactory.get_agent_llm("supervisor")
            >>> # Returns LLM configured with supervisor's provider/model
        """
        if connection is not None:
            # Visitor endpoints reuse the direct-OpenAI per-agent temperatures.
            temperature = Config.AGENT_MODELS_OPENAI.get(agent_name, {}).get(
                "temperature", Config.TEMPERATURE
            )
            return LLMFactory.get_connection_llm(connection, temperature=temperature)

        # Get agent-specific config or fallback to default
        agent_config = Config.AGENT_MODELS().get(
            agent_name, Config.DEFAULT_AGENT_MODEL()
        )

        provider = agent_config.get("provider", Config.MODEL_PROVIDER)
        model = agent_config.get("model", None)  # Get specific model name
        if provider == "azure" and Config.AZURE_OPENAI_DEPLOYMENT_NAME:
            model = Config.AZURE_OPENAI_DEPLOYMENT_NAME
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
