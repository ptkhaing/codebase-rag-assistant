"""Quick manual test for hybrid retrieval.

Usage:
    python -m app.retrieval.hybrid_query "your question here"
"""

import sys

from app.ingestion.vectorstore import VectorStore
from app.retrieval.hybrid import HybridRetriever


def run(query_text: str) -> None:
    store = VectorStore()
    retriever = HybridRetriever(store)
    print("Building BM25 index...")
    retriever.build_index()

    results = retriever.search(query_text, k=5)
    if not results:
        print("No results — did you run ingestion first?")
        return

    for r in results:
        m = r.metadata
        print(
            f"[combined {r.combined_score:.3f}  vec {r.vector_score:.3f}  "
            f"bm25 {r.bm25_score:.3f}] {m.get('file_path')}:{m.get('start_line')}-{m.get('end_line')}"
        )
        print(f"  {r.text.splitlines()[0][:80]}")
        print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python -m app.retrieval.hybrid_query "your question"')
        sys.exit(1)
    run(sys.argv[1])
