"""FastAPI app tying the full RAG pipeline together into a real API.

Endpoints:
  GET  /health   -- liveness check (for Render/UptimeRobot later)
  POST /ingest   -- clone + chunk + embed + store a repo (synchronous;
                    can take a minute or two on first run)
  POST /chat     -- ask a question, get a cited answer
  GET  /eval     -- run the eval harness, return the report

The retriever's BM25 index is built once at startup (if a corpus already
exists from a previous run) and rebuilt after each /ingest call, since
BM25 needs the whole corpus up front and it's cheap enough to just
rebuild rather than incrementally update.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent.loop import run_agent
from app.eval.harness import run_eval
from app.generation.answer import generate_answer
from app.ingestion.chunker import chunk_files
from app.ingestion.loader import clone_and_load
from app.ingestion.vectorstore import VectorStore
from app.retrieval.hybrid import HybridRetriever

app = FastAPI(title="Codebase RAG Assistant")

# Allow the frontend (a different origin in dev, and on Netlify in
# production) to call this API. Tighten allow_origins to your actual
# frontend URL before sharing this publicly -- "*" is fine for local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = VectorStore()
retriever = HybridRetriever(store)


def _rebuild_bm25_if_possible() -> None:
    """BM25 needs the corpus up front. If nothing's been ingested yet,
    this just leaves the retriever empty -- /chat returns a clear error
    instead of crashing."""
    if store.count() > 0:
        retriever.build_index()


_rebuild_bm25_if_possible()  # picks up chroma_data from a previous run, if any


class IngestRequest(BaseModel):
    repo_url: str


class ChatRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": store.count()}


@app.post("/ingest")
def ingest(request: IngestRequest):
    sources = clone_and_load(request.repo_url)
    chunks = chunk_files(sources)
    store.clear()
    store.add_chunks(chunks)
    retriever.build_index()
    return {"files_loaded": len(sources), "chunks_stored": store.count()}


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if store.count() == 0:
        raise HTTPException(status_code=400, detail="No repo ingested yet -- call /ingest first.")

    result = run_agent(retriever, request.query)
    answer = generate_answer(request.query, result.chunks)

    return {
        "answer": answer,
        "rounds_used": result.rounds_used,
        "sources": [
            {
                "file_path": c.metadata.get("file_path"),
                "start_line": c.metadata.get("start_line"),
                "end_line": c.metadata.get("end_line"),
            }
            for c in result.chunks
        ],
    }


@app.get("/eval")
def eval_endpoint():
    report = run_eval()
    return {
        "hit_rate": report["hit_rate"],
        "hits": report["hits"],
        "total": report["total"],
        "failed_questions": [pair.question for pair, hit in report["results"] if not hit],
    }