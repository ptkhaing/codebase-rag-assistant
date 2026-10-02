"""Compares hybrid retrieval results before and after reranking.

Usage:
    python -m app.retrieval.rerank_query "your question here"
"""

import sys

from app.ingestion.vectorstore import VectorStore
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.rerank import rerank


def run(query_text: str) -> None:
    store = VectorStore()
    retriever = HybridRetriever(store)
    print("Building BM25 index...")
    retriever.build_index()

    hybrid_results = retriever.search(query_text, k=10)
    if not hybrid_results:
        print("No results -- did you run ingestion first?")
        return

    print("\n--- BEFORE reranking (hybrid order) ---")
    for r in hybrid_results[:5]:
        m = r.metadata
        print(f"  {m.get('file_path')}:{m.get('start_line')}  (combined={r.combined_score:.3f})")

    print("\nReranking (downloads the reranker model on first run, ~90MB)...")
    reranked = rerank(query_text, hybrid_results, top_k=5)

    print("\n--- AFTER reranking ---")
    for r in reranked:
        m = r.metadata
        print(f"  {m.get('file_path')}:{m.get('start_line')}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python -m app.retrieval.rerank_query "your question"')
        sys.exit(1)
    run(sys.argv[1])