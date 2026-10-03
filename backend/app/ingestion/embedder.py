"""Embeds text using Cohere's hosted Embed API instead of a local model.

Switched from local fastembed (ONNX runtime) because loading an embedding
model into memory, combined with FastAPI + chromadb, exceeded Render's
free-tier 512MB RAM limit. An API call has no local memory footprint for
the model itself -- the trade-off is a network call per request and a
(generous for this project's scale) free-tier rate limit instead of
unlimited local inference.
"""

import os

import cohere
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "embed-english-v3.0"

_client: cohere.Client | None = None


def _get_client() -> cohere.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "COHERE_API_KEY not set. Get a free key from dashboard.cohere.com "
                "and put it in a .env file in the backend/ folder."
            )
        _client = cohere.Client(api_key=api_key)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds texts being STORED (documents) -- Cohere's v3 models need a
    different input_type for stored content vs search queries, for better
    retrieval quality. Use embed_query() for search-time embedding instead."""
    if not texts:
        return []
    client = _get_client()
    response = client.embed(texts=texts, model=MODEL_NAME, input_type="search_document")
    return response.embeddings


def embed_query(text: str) -> list[float]:
    """Embeds a search QUERY, using the query-specific input_type."""
    client = _get_client()
    response = client.embed(texts=[text], model=MODEL_NAME, input_type="search_query")
    return response.embeddings[0]