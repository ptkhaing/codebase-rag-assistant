# Codebase RAG Assistant

A RAG chatbot that answers questions about a codebase with cited, grounded
answers — hybrid retrieval, reranking, an agentic retrieval loop, and a
real evaluation harness, not just an embedding API wrapped in a prompt.

Built specifically to target AI Engineer roles (prompting, retrieval,
agents, evaluation) rather than classical ML Engineer roles — see the
[build guide](./BUILD_GUIDE.md) for why that distinction shaped the scope.

## Live demo

- **Chat UI:** https://codebase-rag-ptk.netlify.app
- **API docs:** https://codebase-rag-assistant-rr30.onrender.com/docs

The demo is pre-loaded with my own [photo-proofing-app](https://github.com/ptkhaing/photo-proofing-app)
repo as its indexed corpus — ask it things like *"how does gallery deletion
work?"* or *"how is the dashboard passcode protected?"*

## Architecture

clone repo → chunk (function/class-aware) → embed (Cohere) → store (Chroma)
│
query → hybrid retrieval (vector + BM25) → rerank (Cohere) → agent loop
(checks if retrieval is sufficient; reformulates + retries once if not)
│
generate cited answer (Groq)

**Stack:** Python/FastAPI backend, React/TypeScript frontend, Chroma
(vector store), Cohere (embeddings + reranking, hosted API), Groq (LLM
generation, hosted API).

## Why hosted APIs instead of local models

The original design ran embeddings and reranking locally via `fastembed`
(ONNX, chosen specifically for being far lighter than `sentence-transformers`/
PyTorch). That still wasn't light enough: loading the model into memory,
combined with FastAPI and Chroma, exceeded Render's free-tier 512MB RAM
limit and crash-looped on every deploy. Switching both embeddings and
reranking to Cohere's hosted API (free tier, no credit card) removed the
local model's memory footprint entirely and fixed it. The trade-off is a
network call per request and a rate-limited free tier instead of
unlimited local inference — a reasonable trade for a portfolio demo's
traffic level.

Render's free tier also wipes disk on every cold start/restart, so the
app auto-re-ingests its demo corpus on startup if the vector store comes
up empty, rather than requiring a manual `/ingest` call first.

## Evaluation

18 hand-written question/expected-file pairs against the real indexed
repo, scoring whether the correct file is retrieved in the top 5 results.

**Result: 94.4% (17/18) retrieval hit-rate.**

The one failure is more nuanced than a clean miss, and worth stating
plainly rather than hiding: for *"how does the app know when a client has
submitted their final photo selections?"*, hybrid search's actual top
result was the correct **file** (`types/photo.ts`) but the wrong **chunk**
within it — the `Photo` interface's field definitions, not the
`GalleryReviewState` union where the `'submitted'` state is actually
defined. Reranking correctly demoted that weak match. But the eval
metric only checks file-level hits, so it can't distinguish "wrong chunk,
right file" from "no hit at all" — it recorded this as a failure even
though reranking was arguably the *correct* call. A chunk-level (not
file-level) eval metric would be the natural refinement here, and is a
known, deliberate scope limitation rather than an oversight.

## Known limitations (stated honestly, not hidden)

- **Chunking is regex-heuristic, not AST-based.** Splits on
  function/class boundaries per language via lightweight regex patterns,
  not a full `tree-sitter` parse. Works well in practice (confirmed by
  the eval above) but a real AST parse would be the production-grade
  version.
- **The agent loop is "agentic-lite."** One extra retrieval round if the
  first pass is judged insufficient, not an open-ended ReAct-style loop
  with arbitrary tool selection.
- **The corpus is currently fixed** to one repo per deployment (set via
  `DEFAULT_REPO_URL`), though `/ingest` accepts any public repo URL at
  runtime. Multi-repo support would be the natural next step.
- **Eval is file-level, not chunk-level** (see above) — a known,
  deliberate simplification, not an oversight.

## Local setup

See [BUILD_GUIDE.md](./BUILD_GUIDE.md) for the full phase-by-phase build
process. Quick start:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# add COHERE_API_KEY and GROQ_API_KEY to a .env file
uvicorn app.main:app --reload
```

```bash
cd rag-frontend
npm install
npm run dev
```