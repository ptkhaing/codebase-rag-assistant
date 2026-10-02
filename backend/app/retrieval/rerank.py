"""Reranks hybrid retrieval results with a cross-encoder for higher
precision. Hybrid retrieval (Phase 2) is optimized for speed over a large
candidate pool; a cross-encoder is slower but scores each (query, chunk)
pair jointly rather than comparing separately-computed embeddings, which
is more accurate. Reranking narrows results down to a final, more
precisely-ordered top_k.
"""

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.retrieval.hybrid import RetrievalResult

MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"

_reranker: TextCrossEncoder | None = None


def _get_reranker() -> TextCrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(model_name=MODEL_NAME)
    return _reranker


def rerank(query: str, results: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
    """Re-scores hybrid results with the cross-encoder and returns the
    top_k, re-sorted by the new score. Leaves each result's original
    vector_score/bm25_score untouched -- only the ordering changes, so
    you can compare before/after."""
    if not results:
        return []

    reranker = _get_reranker()
    documents = [r.text for r in results]
    scores = list(reranker.rerank(query, documents))

    scored = list(zip(results, scores))
    scored.sort(key=lambda pair: -pair[1])

    return [r for r, _ in scored[:top_k]]