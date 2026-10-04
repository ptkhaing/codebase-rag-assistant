"""FastAPI app tying the full RAG pipeline together into a real API.

Endpoints:
  GET  /health   -- liveness check (for Render/UptimeRobot)
  POST /ingest   -- clone + chunk + embed + store a repo (synchronous;
                    can take a minute or two on first run)
  POST /chat     -- ask a question, get a cited answer
  GET  /eval     -- run the eval harness, return the report

On startup, if the vector store is empty (e.g. after a fresh deploy, or
after Render's free tier wipes ephemeral disk on a cold start), the app
auto-ingests a default repo so the live demo self-heals rather than
requiring a manual /ingest call first.
"""

import os

from dotenv import load_dotenv
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

load_dotenv()

DEFAULT_REPO_URL = os.environ.get(
    "DEFAULT_REPO_URL", "https://github.com/ptkhaing/photo-proofing-app.git"
)

app = FastAPI(title="Codebase RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your actual frontend URL once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)

store = VectorStore()
retriever = HybridRetriever(store)


@app.on_event("startup")
def startup_event():
    if store.count() == 0:
        print(f"Empty vector store on startup -- auto-ingesting {DEFAULT_REPO_URL}")
        sources = clone_and_load(DEFAULT_REPO_URL)
        chunks = chunk_files(sources)
        store.add_chunks(chunks)
    retriever.build_index()


class IngestRequest(BaseModel):
    repo_url: str


class ChatRequest(BaseModel):
    query: str


@app.api_route("/health", methods=["GET", "HEAD"])
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