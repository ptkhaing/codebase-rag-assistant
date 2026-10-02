"""Eval harness: measures retrieval hit-rate against the hand-written
QA_PAIRS set. For each question, checks whether the expected file appears
anywhere in the top-k retrieved+reranked results -- measuring whether
retrieval found the right source material, not whether the final
generated answer text is perfect (a separate, fuzzier thing to measure).
"""

from app.eval.qa_pairs import QA_PAIRS, QAPair
from app.ingestion.vectorstore import VectorStore
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.rerank import rerank


def _hit(pair: QAPair, retriever: HybridRetriever, k: int) -> bool:
    hybrid_results = retriever.search(pair.question, k=10)
    reranked = rerank(pair.question, hybrid_results, top_k=k)
    return any(pair.expected_file in r.metadata.get("file_path", "") for r in reranked)


def run_eval(k: int = 5) -> dict:
    store = VectorStore()
    retriever = HybridRetriever(store)
    retriever.build_index()

    results = []
    for pair in QA_PAIRS:
        hit = _hit(pair, retriever, k)
        results.append((pair, hit))

    hits = sum(1 for _, hit in results if hit)
    total = len(results)
    hit_rate = hits / total if total else 0.0

    return {"hit_rate": hit_rate, "hits": hits, "total": total, "results": results}


def print_report(k: int = 5) -> None:
    report = run_eval(k)
    print(f"Retrieval hit-rate @ k={k}: {report['hits']}/{report['total']} ({report['hit_rate']:.1%})\n")

    failures = [pair for pair, hit in report["results"] if not hit]
    if failures:
        print("Failed questions:")
        for pair in failures:
            print(f"  - {pair.question}  (expected: {pair.expected_file})")
    else:
        print("All questions passed.")


if __name__ == "__main__":
    print_report()