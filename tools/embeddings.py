# ==========================================================================
# Master Thesis - Query Embeddings Backends
#   - André Filipe Gomes Silvestre, 20240502
#
#   Builds the embeddings object used by the ChromaDB knowledge base.
#   Features:
#     - 'local': BAAI/bge-m3 through sentence-transformers (torch)
#     - 'cloudflare': the same model on Cloudflare Workers AI, L2-normalised
#       so distances match the locally built store
#     - One startup log line naming the active backend
# ==========================================================================

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from typing import List

from langchain_core.embeddings import Embeddings

from config import Config


# Source: https://developers.cloudflare.com/workers-ai/models/bge-m3/
CLOUDFLARE_API_BASE_URL = "https://api.cloudflare.com/client/v4"
CLOUDFLARE_MAX_BATCH_SIZE = 100
CLOUDFLARE_TIMEOUT_SECONDS = 20
CLOUDFLARE_RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
CLOUDFLARE_RETRY_DELAY_SECONDS = 1.0
# One search embeds the same query once per collection; caching keeps that to
# one remote call.
QUERY_CACHE_SIZE = 256


class CloudflareEmbeddingsError(RuntimeError):
    """Raised when Cloudflare Workers AI does not return usable embeddings."""


def _l2_normalise(vector: List[float]) -> List[float]:
    """Return ``vector`` scaled to unit length.

    The ChromaDB store is built with ``normalize_embeddings=True``, so query
    vectors must have unit length for the L2 distance thresholds to hold.
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise CloudflareEmbeddingsError("Cloudflare Workers AI returned a zero vector.")
    return [value / norm for value in vector]


class CloudflareWorkersAIEmbeddings(Embeddings):
    """BAAI/bge-m3 embeddings computed by Cloudflare Workers AI.

    Errors name the HTTP status only, never the account ID or the token.
    """

    def __init__(self, account_id: str, api_token: str, model: str = Config.CLOUDFLARE_EMBEDDING_MODEL):
        """Create a client for the Workers AI REST endpoint.

        Args:
            account_id: Cloudflare account ID.
            api_token: API token with the "Workers AI: Read" permission.
            model: Workers AI model name.
        """
        import httpx

        self._httpx = httpx
        self._url = f"{CLOUDFLARE_API_BASE_URL}/accounts/{account_id}/ai/run/{model}"
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=CLOUDFLARE_TIMEOUT_SECONDS,
        )
        self._query_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._query_cache_lock = threading.Lock()

    def _post(self, payload: dict):
        """POST one request; network errors are reported without the URL."""
        try:
            return self._client.post(self._url, json=payload)
        except self._httpx.HTTPError as exc:
            raise CloudflareEmbeddingsError(
                f"Cloudflare Workers AI request failed ({type(exc).__name__})."
            ) from None

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed up to ``CLOUDFLARE_MAX_BATCH_SIZE`` texts, retrying once on 429 and 5xx."""
        # Workers AI rejects empty strings (minLength 1).
        payload = {"text": [text if text.strip() else " " for text in texts], "truncate_inputs": True}
        response = self._post(payload)
        if response.status_code in CLOUDFLARE_RETRY_STATUS_CODES:
            time.sleep(CLOUDFLARE_RETRY_DELAY_SECONDS)
            response = self._post(payload)
        if response.status_code != 200:
            raise CloudflareEmbeddingsError(f"Cloudflare Workers AI returned HTTP {response.status_code}.")
        body = response.json()
        data = (body.get("result") or {}).get("data")
        if not body.get("success", False) or not isinstance(data, list) or len(data) != len(texts):
            raise CloudflareEmbeddingsError("Cloudflare Workers AI returned an unexpected embeddings payload.")
        return [_l2_normalise(vector) for vector in data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents in batches accepted by Workers AI."""
        vectors: List[List[float]] = []
        for start in range(0, len(texts), CLOUDFLARE_MAX_BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[start:start + CLOUDFLARE_MAX_BATCH_SIZE]))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed one query, reusing recent results."""
        with self._query_cache_lock:
            cached = self._query_cache.get(text)
            if cached is not None:
                self._query_cache.move_to_end(text)
                return cached
        vector = self._embed_batch([text])[0]
        with self._query_cache_lock:
            self._query_cache[text] = vector
            if len(self._query_cache) > QUERY_CACHE_SIZE:
                self._query_cache.popitem(last=False)
        return vector


def _create_local_embeddings(use_gpu: bool) -> Embeddings:
    """Load BAAI/bge-m3 with sentence-transformers on CUDA or CPU."""
    try:
        import torch
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError(
            "The local embeddings backend needs requirements_local_embeddings.txt "
            "(torch, sentence-transformers). Hosted deployments set "
            "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN instead."
        ) from exc

    # The hosted Space installs CPU-only torch; local CUDA builds use the GPU.
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"🧭 Embeddings backend: local ({Config.EMBEDDING_MODEL_NAME} on {device.upper()})", flush=True)
    try:
        return HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        if device == "cpu":
            raise
        print(f"   ⚠️ GPU error: {exc}. Falling back to CPU.", flush=True)
        return HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


def create_embeddings(use_gpu: bool = True, backend: str | None = None) -> Embeddings:
    """Create the embeddings object for the knowledge base.

    Args:
        use_gpu: Whether the local backend may use CUDA.
        backend: ``local`` or ``cloudflare``; defaults to
            ``Config.embeddings_backend()``. Building the store always uses
            ``local``.

    Returns:
        Embeddings: An object with ``embed_query`` and ``embed_documents``.
    """
    resolved_backend = backend or Config.embeddings_backend()
    if resolved_backend == "cloudflare":
        if not (Config.CLOUDFLARE_ACCOUNT_ID and Config.CLOUDFLARE_API_TOKEN):
            raise CloudflareEmbeddingsError(
                "The cloudflare embeddings backend needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN."
            )
        print(
            f"🧭 Embeddings backend: cloudflare ({Config.CLOUDFLARE_EMBEDDING_MODEL} on Workers AI)",
            flush=True,
        )
        return CloudflareWorkersAIEmbeddings(Config.CLOUDFLARE_ACCOUNT_ID, Config.CLOUDFLARE_API_TOKEN)
    if resolved_backend == "local":
        return _create_local_embeddings(use_gpu)
    raise ValueError(f"Unknown embeddings backend: {resolved_backend!r} (expected 'local' or 'cloudflare').")
