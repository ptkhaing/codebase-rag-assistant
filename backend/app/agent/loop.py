"""Agentic-lite retrieval loop: retrieve, check if the result is
sufficient to answer the question, and if not, reformulate the query
once and retrieve again. Capped at one extra round -- a deliberate v1
scope decision (an open-ended ReAct-style loop is a reasonable future
extension, not built here).
"""

from dataclasses import dataclass

from app.generation.llm import chat
from app.retrieval.hybrid import HybridRetriever, RetrievalResult
from app.retrieval.rerank import rerank


@dataclass
class AgentResult:
    chunks: list[RetrievalResult]
    rounds_used: int
    reformulated_query: str | None  # None if only one round was needed


def _is_sufficient(query: str, chunks: list[RetrievalResult]) -> bool:
    context = "\n\n".join(f"[{c.metadata.get('file_path')}]\n{c.text}" for c in chunks)
    prompt = (
        f"Question: {query}\n\n"
        f"Retrieved code chunks:\n{context}\n\n"
        "Can these chunks fully answer the question? Reply with exactly "
        "one word: yes or no."
    )
    answer = chat([{"role": "user", "content": prompt}]).strip().lower()
    return answer.startswith("yes")


def _reformulate_query(original_query: str, chunks: list[RetrievalResult]) -> str:
    context = "\n\n".join(f"[{c.metadata.get('file_path')}]" for c in chunks)
    prompt = (
        f"Original question: {original_query}\n\n"
        f"These files were retrieved but weren't sufficient to answer it:\n{context}\n\n"
        "Suggest a better, more specific search query (different wording or "
        "more specific terms) that might find the missing information. "
        "Reply with ONLY the new search query, nothing else."
    )
    return chat([{"role": "user", "content": prompt}]).strip()


def run_agent(retriever: HybridRetriever, query: str, k: int = 5) -> AgentResult:
    hybrid_results = retriever.search(query, k=10)
    chunks = rerank(query, hybrid_results, top_k=k)

    if not chunks or _is_sufficient(query, chunks):
        return AgentResult(chunks=chunks, rounds_used=1, reformulated_query=None)

    new_query = _reformulate_query(query, chunks)
    second_hybrid = retriever.search(new_query, k=10)
    second_chunks = rerank(new_query, second_hybrid, top_k=k)

    merged_by_id: dict[str, RetrievalResult] = {}
    for c in chunks + second_chunks:
        existing = merged_by_id.get(c.id)
        if existing is None or c.combined_score > existing.combined_score:
            merged_by_id[c.id] = c

    merged = sorted(merged_by_id.values(), key=lambda c: -c.combined_score)[:k]

    return AgentResult(chunks=merged, rounds_used=2, reformulated_query=new_query)