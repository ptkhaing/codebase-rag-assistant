"""Runs the full ingestion pipeline: clone -> chunk -> embed -> store.

Usage:
    python -m app.ingestion.pipeline <repo_url>
"""

import sys
import time

from app.ingestion.chunker import chunk_files
from app.ingestion.loader import clone_and_load
from app.ingestion.vectorstore import VectorStore


def run(repo_url: str) -> None:
    print(f"Cloning and loading {repo_url}...")
    t0 = time.time()
    sources = clone_and_load(repo_url)
    print(f"  Loaded {len(sources)} files ({time.time() - t0:.1f}s)")

    t0 = time.time()
    chunks = chunk_files(sources)
    print(f"  Produced {len(chunks)} chunks ({time.time() - t0:.1f}s)")

    print("Embedding and storing (this downloads the model on first run)...")
    t0 = time.time()
    store = VectorStore()
    store.clear()  # avoid accumulating duplicates on repeated ingestion runs
    store.add_chunks(chunks)
    print(f"  Stored {store.count()} chunks ({time.time() - t0:.1f}s)")

    print("\nDone. Try a query:")
    print('  python -m app.ingestion.pipeline_query "how does gallery deletion work?"')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m app.ingestion.pipeline <repo_url>")
        sys.exit(1)
    run(sys.argv[1])
