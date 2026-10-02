"""Quick manual sanity-check for the vector store after ingestion.

Usage:
    python -m app.ingestion.pipeline_query "your question here"
"""

import sys

from app.ingestion.vectorstore import VectorStore


def run(query_text: str) -> None:
    store = VectorStore()
    print(f"Collection has {store.count()} chunks total.\n")
    results = store.query(query_text, k=5)
    if not results:
        print("No results — did you run the ingestion pipeline first?")
        return
    for r in results:
        m = r["metadata"]
        print(f"[{r['distance']:.3f}] {m['file_path']}:{m['start_line']}-{m['end_line']}")
        print(f"  {r['text'].splitlines()[0][:80]}")
        print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python -m app.ingestion.pipeline_query "your question"')
        sys.exit(1)
    run(sys.argv[1])
