"""Tests the full agent loop: retrieve -> check sufficiency -> maybe
retrieve again -> return final chunks.

Usage:
    python -m app.agent.agent_query "your question here"
"""

import sys

from app.agent.loop import run_agent
from app.ingestion.vectorstore import VectorStore
from app.retrieval.hybrid import HybridRetriever


def run(query_text: str) -> None:
    store = VectorStore()
    retriever = HybridRetriever(store)
    print("Building BM25 index...")
    retriever.build_index()

    result = run_agent(retriever, query_text)

    print(f"\nRounds used: {result.rounds_used}")
    if result.reformulated_query:
        print(f"Reformulated query: {result.reformulated_query!r}")

    print("\nFinal chunks:")
    for c in result.chunks:
        m = c.metadata
        print(f"  {m.get('file_path')}:{m.get('start_line')}-{m.get('end_line')}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python -m app.agent.agent_query "your question"')
        sys.exit(1)
    run(sys.argv[1])