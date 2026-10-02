# Code-Repo RAG Assistant — Build Guide

A RAG chatbot that answers questions about a codebase (starting with your
own `photo-proofing-app` repo) with cited code snippets. Targets AI
Engineer roles specifically — retrieval + reranking + agentic loop + a real
eval harness, not just "call an embedding API and paste results into a
prompt."

Free-tier stack throughout: **fastembed** (ONNX, not torch — much lighter,
fits Render's free 512MB RAM), **Chroma** (vector store), **rank-bm25**
(keyword search), **Groq free tier** (LLM generation).

---

## Phase 0 — Already done for you

`app/ingestion/loader.py` and `app/ingestion/chunker.py` are built and
tested against your real `photo-proofing-app` GitHub repo — 25 files
loaded, 102 chunks produced, verified splitting cleanly on real
function/component boundaries (`export function PhotoGrid(...)`, etc.),
not blind character counts. Use these as-is; no need to rebuild them.

**Known limitation, on purpose:** chunking uses regex heuristics per
language (Python `def`/`class`, JS/TS function/const/interface patterns),
not a full AST parse. A real upgrade path is `tree-sitter`, worth
mentioning as a stretch goal in your README rather than building now.

---

## Phase 1 — Embedding + vector store

**Goal:** turn chunks into vectors, store them queryably.

1. `pip install fastembed chromadb`
2. Write `app/ingestion/embedder.py`:
   - Wrap `fastembed.TextEmbedding(model_name="BAAI/bge-small-en-v1.5")`
   - `embed_texts(texts: list[str]) -> list[list[float]]`
3. **First run will download the model (~130MB) from HuggingFace** — this
   needs real internet access. It'll work fine on your machine/Render; it
   just couldn't run in my sandbox (network allowlist blocked
   huggingface.co there specifically). Confirm it downloads and embeds
   successfully on your end before moving on.
4. Write `app/ingestion/vectorstore.py`:
   - `chromadb.PersistentClient(path="./chroma_data")`
   - One collection, e.g. `code_chunks`
   - Store each chunk's embedding + metadata (`file_path`, `start_line`,
     `end_line`, `language`, and the raw `text` itself so you can retrieve
     it without a second lookup)
5. Write `app/ingestion/pipeline.py` tying loader → chunker → embedder →
   vectorstore together, callable as a script:
   `python -m app.ingestion.pipeline <repo_url>`
6. **Test:** run it against your photo-proofing-app repo, confirm the
   Chroma collection has ~102 entries (matching the chunk count we already
   verified), and manually query it once with `collection.query(...)` to
   sanity-check a result makes sense before building anything on top.

---

## Phase 2 — Hybrid retrieval (vector + BM25)

**Goal:** combine semantic search with exact keyword matching. Vector
search alone often misses exact symbol names (`useGalleryStore`,
`PasscodeGate`) — BM25 catches those.

1. `pip install rank-bm25`
2. Write `app/retrieval/hybrid.py`:
   - Build a `BM25Okapi` index over all chunk texts (tokenize on
     whitespace/camelCase-aware split — simple `.split()` is fine to start)
   - `vector_search(query, k=10)` → top-k from Chroma
   - `bm25_search(query, k=10)` → top-k from BM25
   - `hybrid_search(query, k=10)`: merge both result sets, dedupe by chunk
     id, combine scores (simplest: normalize each score set to 0-1, sum
     them — reciprocal rank fusion is the "more correct" version if you
     want to go further)
3. **Test:** run a query containing an exact symbol name (e.g.
   `"PasscodeGate"`) and confirm BM25 surfaces it even when the vector
   score alone wouldn't rank it top.

---

## Phase 3 — Reranking

**Goal:** re-score the hybrid results with a more accurate (but slower)
model, since retrieval's first pass optimizes for speed over precision.

1. `fastembed` also ships `TextCrossEncoder` — no new heavy dependency
   needed. Model: `"Xenova/ms-marco-MiniLM-L-6-v2"` or similar small
   cross-encoder.
2. Write `app/retrieval/rerank.py`:
   - Take the ~15-20 hybrid results, cross-encode each against the query,
     re-sort by that score, keep top 5.
3. **Test:** compare top-5-before-rerank vs top-5-after-rerank on a
   sample query — you should see the order change, ideally the most
   relevant chunk moving up.

---

## Phase 4 — Agent loop (retrieve → check sufficiency → retrieve again)

**Goal:** don't just always do one retrieve-then-answer pass — decide if
the first retrieval was enough.

1. Write `app/agent/loop.py`:
   - Retrieve (hybrid + rerank) for the query
   - Ask the LLM (cheap, single call): "Given these chunks, can you fully
     answer the question? yes/no" — a small structured prompt, not a full
     generation
   - If no: reformulate the query once (e.g. ask the LLM to produce a
     better search query given what's missing) and retrieve again, merge
     with the first results
   - Cap at one extra retrieval round — don't let it loop indefinitely
2. This is intentionally "agentic-lite," not a full ReAct loop with
   arbitrary tool selection — that's a reasonable v1 scope, and you can
   describe it honestly as that in interviews.

---

## Phase 5 — Generation (Groq)

**Goal:** produce the final cited answer.

1. `pip install groq`, get a free API key from console.groq.com
2. Write `app/generation/llm.py`:
   - Model: something like `llama-3.1-8b-instant` (check Groq's current
     free-tier model list — it changes)
   - Prompt: system message instructing it to answer *only* from the
     provided chunks, cite `file_path:start_line-end_line` for each claim,
     and say "I don't know" if the chunks don't cover it
3. **Never commit the API key.** Use `.env` + `python-dotenv`, same
   pattern as your other projects' env var handling.
4. **Test:** ask a real question about your repo ("how does gallery
   deletion work?") and confirm the answer cites actual files that contain
   the relevant code.

---

## Phase 6 — Eval harness

**Goal:** the piece most junior portfolios skip. This is what actually
signals "I understand how to validate an AI system," not just build one.

1. Write `app/eval/qa_pairs.py`: hand-write 15-20 question/expected-answer
   pairs about your own repo, where you *know* which file(s) should be
   retrieved for each (e.g. "How does the passcode gate work?" → should
   retrieve `PasscodeGate.tsx`).
2. Write `app/eval/harness.py`:
   - For each pair: run retrieval, check if the expected file is in the
     top-k results (retrieval hit-rate)
   - Optionally: run full generation and score answer quality with a
     simple LLM-as-judge prompt ("does this answer correctly address the
     question, given the reference answer? yes/no")
   - Output a simple report: hit-rate %, and which questions failed
3. **This is your best interview talking point from the whole project** —
   "I built an eval set and measured X% retrieval accuracy" is exactly
   what separates this from a tutorial clone.

---

## Phase 7 — FastAPI wiring

1. `app/main.py`: endpoints —
   - `POST /ingest` (repo_url) → runs the pipeline, kicks off indexing
   - `POST /chat` (query) → runs the full agent loop, returns answer +
     citations
   - `GET /eval` → runs the harness, returns the report
2. Handle the reality that ingestion takes a while (repo clone + embed) —
   either run it synchronously and accept a slow request, or return
   immediately and let the client poll a status endpoint. Synchronous is
   fine for v1 given the scope.

---

## Phase 8 — Frontend (React + TypeScript)

Same stack/quality bar as your photo-proofing-app:

1. Vite + React + TS (strict mode, matching your existing standard)
2. A simple chat UI: message list, input box, each assistant message
   renders its citations as clickable/expandable code snippets
3. Call your FastAPI backend's `/chat` endpoint
4. Optional but a nice touch: a small "sources used" panel showing which
   chunks were retrieved, not just the final answer — makes the
   retrieval step visible, which is more interesting to a technical
   reviewer than a plain chat box

---

## Phase 9 — Deploy

1. Backend → Render (same pattern as your other two Python/Java
   projects). **Watch the free-tier RAM ceiling** — this is exactly why
   fastembed over sentence-transformers mattered.
2. Add the same `/health` endpoint pattern you already added to
   multitenant-saas-backend, and set up an UptimeRobot monitor the same
   way.
3. Frontend → Netlify, same as photo-proofing-app. Don't forget
   `public/_redirects` for SPA routing — you already hit this exact bug
   once, so it's a known gotcha now.
4. Env vars (Groq API key, etc.) → set in Render's environment variables
   dashboard, never committed — same pattern as the `VITE_DASHBOARD_PASSCODE`
   fix from photo-proofing-app.

---

## What to write in the README when you're done

Be upfront about scope, the same way the photo-proofing-app README is:
- Chunking is regex-heuristic, not AST-based (tree-sitter = real upgrade)
- Agent loop is "agentic-lite" (one extra retrieval round, not open-ended)
- Corpus is currently fixed to your own repo for the demo; multi-repo
  support would be the natural next step
- Eval hit-rate number, stated plainly, with the actual number — not
  just "I built an eval harness"
