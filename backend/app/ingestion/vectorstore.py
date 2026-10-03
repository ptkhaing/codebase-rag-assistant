"""Persistent Chroma vector store for code chunks.

Stores each chunk's embedding alongside metadata (file path, line range,
language) and the raw chunk text itself, so a retrieval hit doesn't need
a second lookup to get the actual content back.
"""

from pathlib import Path

import chromadb

from app.ingestion.chunker import Chunk
from app.ingestion.embedder import embed_query, embed_texts

DEFAULT_DB_PATH = str(Path(__file__).resolve().parents[2] / "chroma_data")
COLLECTION_NAME = "code_chunks"


class VectorStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)

    def clear(self) -> None:
        """Wipes the collection — useful when re-ingesting a repo from
        scratch rather than accumulating duplicate/stale chunks."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 16) -> None:
        """Embeds and stores chunks in batches (embedding many texts at
        once is much faster than one-by-one)."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embeddings = embed_texts([c.text for c in batch])
            self._collection.add(
                ids=[c.id for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "file_path": c.file_path,
                        "language": c.language,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                    }
                    for c in batch
                ],
            )

    def query(self, query_text: str, k: int = 10) -> list[dict]:
        """Returns up to k results, each a dict with id, text, metadata,
        and distance (lower = more similar)."""
        query_embedding = embed_query(query_text)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )
        if not results["ids"] or not results["ids"][0]:
            return []
        return [
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

    def count(self) -> int:
        return self._collection.count()

    def all_chunks(self) -> list[dict]:
        """Returns every stored chunk as {id, text, metadata} — used by
        BM25, which needs the full corpus (text) up front to build its
        keyword index, and by hybrid retrieval to fill in metadata for
        chunks that only matched via BM25 (not vector search)."""
        results = self._collection.get()
        return [
            {"id": results["ids"][i], "text": results["documents"][i], "metadata": results["metadatas"][i]}
            for i in range(len(results["ids"]))
        ]
