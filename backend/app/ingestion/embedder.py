"""Embeds text using fastembed (ONNX-based, no torch dependency).

Deliberately not sentence-transformers: that pulls in PyTorch, which is a
several-hundred-MB install and needs more RAM than Render's free tier
(512MB) comfortably provides. fastembed's ONNX runtime is lighter on both
disk and memory, at the cost of a slightly smaller model selection.

NOTE: the first call downloads the model (~130MB) from HuggingFace. This
needs normal internet access — it will NOT work in a network-sandboxed
environment, only on a real machine/server.
"""

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Returns one embedding vector per input text, same order."""
    if not texts:
        return []
    model = _get_model()
    return [vec.tolist() for vec in model.embed(texts)]


def embed_query(text: str) -> list[float]:
    """Convenience wrapper for embedding a single query string."""
    return embed_texts([text])[0]
