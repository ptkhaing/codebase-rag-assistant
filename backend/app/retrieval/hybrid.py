"""Hybrid retrieval: combines dense vector search (semantic similarity)
with BM25 keyword search (exact term matching).

Why both: vector search alone often misses exact symbol names. A query
containing "PasscodeGate" isn't guaranteed to rank a chunk defining
PasscodeGate highly by embedding similarity alone -- semantic similarity
and exact lexical identity aren't the same thing. BM25 catches that exact
match directly. Conversely, BM25 alone misses queries phrased in natural
language with no shared vocabulary with the code ("how does deletion
work?" sharing zero exact tokens with a function named `reducer`). Neither
alone is enough; combining them covers both cases.
"""

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.ingestion.vectorstore import VectorStore

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Splits on camelCase/snake_case/non-alphanumeric boundaries and
    lowercases everything, so a query for "gallery store" matches a
    symbol named galleryStore, GalleryStore, or gallery_store alike."""
    text = _CAMEL_BOUNDARY.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)
    return [t.lower() for t in text.split() if t]


@dataclass
class RetrievalResult:
    id: str
    text: str
    metadata: dict
    vector_score: float  # normalized 0-1, higher = more similar
    bm25_score: float  # normalized 0-1, higher = more similar

    @property
    def combined_score(self) -> float:
        return self.vector_score + self.bm25_score


class HybridRetriever:
    def __init__(self, store: VectorStore):
        self.store = store
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._chunk_by_id: dict[str, dict] = {}

    def build_index(self) -> None:
        """Builds the BM25 index over every chunk currently in the vector
        store. Must be called once after ingestion, before searching --
        unlike vector search (which queries the store live per-request),
        BM25 needs the whole corpus up front to compute term statistics."""
        chunks = self.store.all_chunks()
        self._chunk_ids = [c["id"] for c in chunks]
        self._chunk_by_id = {c["id"]: c for c in chunks}
        tokenized_corpus = [tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def _vector_search(self, query: str, k: int) -> dict[str, tuple[float, dict, str]]:
        results = self.store.query(query, k=k)
        if not results:
            return {}
        distances = [r["distance"] for r in results]
        min_d, max_d = min(distances), max(distances)
        span = (max_d - min_d) or 1.0
        out = {}
        for r in results:
            # distance: lower = more similar. Invert + normalize to 0-1
            # so it's directly comparable/summable with the BM25 score.
            similarity = 1 - (r["distance"] - min_d) / span
            out[r["id"]] = (similarity, r["metadata"], r["text"])
        return out

    def _bm25_search(self, query: str, k: int) -> dict[str, float]:
        if self._bm25 is None:
            raise RuntimeError("Call build_index() before searching.")
        scores = self._bm25.get_scores(tokenize(query))
        if len(scores) == 0:
            return {}
        max_score = max(scores) or 1.0
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda pair: -pair[1])[:k]
        return {cid: score / max_score for cid, score in ranked if score > 0}

    def search(self, query: str, k: int = 5, candidate_k: int = 20) -> list[RetrievalResult]:
        """Searches a wider candidate pool (candidate_k) with both methods,
        merges by normalized combined score, returns the top k."""
        vector_results = self._vector_search(query, candidate_k)
        bm25_scores = self._bm25_search(query, candidate_k)

        all_ids = set(vector_results) | set(bm25_scores)
        merged: list[RetrievalResult] = []
        for cid in all_ids:
            vec_score, metadata, text = vector_results.get(cid, (0.0, None, None))
            if metadata is None:
                # Matched only via BM25 -- pull metadata/text from the
                # cached chunk map built in build_index() instead.
                chunk = self._chunk_by_id.get(cid, {})
                metadata = chunk.get("metadata", {})
                text = chunk.get("text", "")
            merged.append(
                RetrievalResult(
                    id=cid,
                    text=text,
                    metadata=metadata,
                    vector_score=vec_score,
                    bm25_score=bm25_scores.get(cid, 0.0),
                )
            )

        merged.sort(key=lambda r: -r.combined_score)
        return merged[:k]
