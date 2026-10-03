"""Reranks hybrid retrieval results using Cohere's hosted Rerank API
instead of a local cross-encoder model -- same memory reason as
embedder.py, removing a second local ONNX model from the deployed
service's footprint.
"""

import os

import cohere
from dotenv import load_dotenv

from app.retrieval.hybrid import RetrievalResult

load_dotenv()

MODEL_NAME = "rerank-english-v3.0"

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


def rerank(query: str, results: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
    if not results:
        return []

    client = _get_client()
    documents = [r.text for r in results]
    response = client.rerank(model=MODEL_NAME, query=query, documents=documents, top_n=top_k)

    return [results[r.index] for r in response.results]