"""Full end-to-end test: agent retrieval -> cited answer generation.

Usage:
    python -m app.generation.answer_query "your question here"
"""

import sys

from app.agent.loop import run_agent
from app.generation.answer import generate_answer
from app.ingestion.vectorstore import VectorStore
from app.retrieval.hybrid import HybridRetriever


def run(query_text: str) -> None:
    store = VectorStore()
    retriever = HybridRetriever(store)
    print("Building BM25 index...")
    retriever.build_index()

    print("Retrieving...")
    result = run_agent(retriever, query_text)
    print(f"(used {result.rounds_used} retrieval round(s))\n")

    print("Generating answer...")
    answer = generate_answer(query_text, result.chunks)

    print("\n--- ANSWER ---")
    print(answer)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python -m app.generation.answer_query "your question"')
        sys.exit(1)
    run(sys.argv[1])