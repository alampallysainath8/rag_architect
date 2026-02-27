# PageIndex RAG

A **vectorless** Retrieval-Augmented Generation system that replaces the traditional chunk-embed-search pipeline with a hierarchical document tree navigated by an LLM.

---

## Table of Contents

1. [What is PageIndex RAG?](#what-is-pageindex-rag)
2. [How It Differs from Traditional RAG](#how-it-differs-from-traditional-rag)
3. [Architecture Overview](#architecture-overview)
4. [Project Structure](#project-structure)
5. [Setup](#setup)
6. [Running the API](#running-the-api)
7. [API Reference](#api-reference)
8. [Avoiding Re-indexing](#avoiding-re-indexing)
9. [Frontend](#frontend)

---

## What is PageIndex RAG?

PageIndex RAG uses the [PageIndex](https://pageindex.ai) service to pre-process a PDF into a **structured hierarchical tree** — a multi-level outline where every node has a title, a summary, and the full text for that section. Instead of retrieving text through vector similarity, an LLM reads the tree's titles and summaries, reasons about which nodes are relevant to the user's question, and then retrieves only those exact sections as context.

The result is a pipeline with:

- **Zero embedding models** — no sentence-transformers, no OpenAI embeddings, nothing local to host.
- **Zero vector databases** — no Pinecone, ChromaDB, Weaviate, or FAISS clusters to maintain.
- **Semantic precision** — the LLM selects nodes by understanding the document's logical structure, not by cosine distance.

---

## How It Differs from Traditional RAG

| Dimension | Traditional RAG | PageIndex RAG |
|---|---|---|
| **Preprocessing** | Split document into fixed-size or semantic chunks | PageIndex builds a hierarchical tree (title + summary + text per node) |
| **Embedding** | Required — every chunk is embedded with a model | Not required — no embedding model at all |
| **Storage** | Vector store (Pinecone, ChromaDB, FAISS …) | PageIndex cloud + a local `tree.json` cache |
| **Retrieval** | ANN search over embedding vectors | LLM reads tree summaries, selects node IDs |
| **Context quality** | Can mix semantically-close-but-irrelevant chunks | Retrieves complete, logically-bounded document sections |
| **Re-indexing cost** | Re-embed every changed chunk; update vector records | Only re-upload if the file is genuinely new (cache handles the rest) |
| **Infrastructure** | Embedding model inference + vector DB | PageIndex API + any fast LLM (Groq used here) |

### Retrieval Flow Comparison

**Traditional RAG**

```
PDF → Chunker → Embedding Model → Vector DB ──ANN query──► Top-K chunks → LLM → Answer
```

**PageIndex RAG**

```
PDF → PageIndex API → Hierarchical Tree
                            │
              User question ▼
              LLM reads tree (titles + summaries)
              LLM selects relevant node IDs
                            │
              Fetch full text of selected nodes
                            │
              LLM answers using that context → Answer
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  Streamlit Frontend              │
│  Upload PDF │ Chat │ Tree Explorer │ Doc List    │
└──────────────────────┬──────────────────────────┘
                       │ HTTP (port 8001)
┌──────────────────────▼──────────────────────────┐
│                  FastAPI Backend  (api.py)       │
│                                                  │
│  /upload  ──► doc_id_cache.json  ──► PageIndex  │
│  /status  ──► PageIndex API                      │
│  /tree    ──► output/<doc_id>/tree.json (cache) │
│  /query   ──► Tree search (Groq LLM)            │
│               Context extraction                 │
│               Answer generation (Groq LLM)      │
│  /documents ──► cache + PageIndex API            │
│  /pdf     ──► docs/ folder                       │
└──────────────────────────────────────────────────┘
          │                        │
   PageIndex API             Groq API
  (tree building)          (LLM inference)
```

---

## Project Structure

```
rag_pageindex/
├── api.py                  # FastAPI backend — all HTTP endpoints
├── pageindex_rag.py        # Standalone script (single-query demo)
├── main.py                 # Entry-point placeholder
├── pyproject.toml          # Project dependencies
├── doc_id_cache.json       # Auto-generated: filename → doc_id map
├── docs/                   # Uploaded PDFs are stored here
├── output/                 # Per-document cached results
│   └── <doc_id>/
│       ├── tree.json       # Cached document tree
│       ├── node_search.json# Last node-selection result
│       └── answer.json     # Last answer
└── frontend/
    ├── app.py              # Streamlit UI
    └── requirements.txt
```

---

## Setup

### 1. Prerequisites

- Python 3.11+
- A [PageIndex](https://pageindex.ai) account and API key
- A [Groq](https://console.groq.com) account and API key

### 2. Install backend dependencies

```bash
cd rag_pageindex
pip install -e .
```

Or with pip directly:

```bash
pip install pageindex groq fastapi "uvicorn[standard]" python-multipart requests python-dotenv
```

### 3. Configure environment variables

Create a `.env` file in `rag_pageindex/`:

```env
PAGEINDEX_API_KEY=your_pageindex_key_here
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile   # optional, this is the default
```

### 4. Install frontend dependencies

```bash
cd frontend
pip install streamlit requests
```

---

## Running the API

Start the FastAPI backend from the `rag_pageindex/` directory:

```bash
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

The API will be available at `http://127.0.0.1:8001`.

Interactive docs (Swagger UI) are auto-generated at:

```
http://127.0.0.1:8001/docs
```

Start the Streamlit frontend (separate terminal):

```bash
cd frontend
streamlit run app.py
```

---

## API Reference

### `GET /health`

Health check. Returns key presence status.

```json
{ "status": "ok", "pageindex_key": true, "groq_key": true }
```

---

### `POST /upload`

Upload a PDF to index. The backend saves the file locally, checks the cache and PageIndex for an existing index, and only submits a new document if none is found.

**Request:** `multipart/form-data` with a `file` field (PDF only).

**Response:**

```json
{
  "doc_id": "pi-xxxxxxxxxxxx",
  "filename": "paper.pdf",
  "status": "processing"
}
```

`status` will be `"processing"` or `"completed"`. Poll `/status/{doc_id}` to wait for completion.

---

### `GET /status/{doc_id}`

Check the processing status of a submitted document.

**Response:**

```json
{
  "doc_id": "pi-xxxxxxxxxxxx",
  "status": "completed",
  "ready": true,
  "meta": { ... }
}
```

Poll this endpoint (e.g. every 5 seconds) until `"ready": true` before querying.

---

### `GET /tree/{doc_id}`

Fetch the full hierarchical document tree. Returns a locally cached version if available, otherwise fetches from PageIndex and caches it.

**Response:** The raw tree JSON with nodes containing `id`, `title`, `summary`, and `text`.

---

### `POST /query`

Run a full RAG cycle against an indexed document.

**Request body:**

```json
{
  "doc_id": "pi-xxxxxxxxxxxx",
  "question": "What are the main conclusions of this paper?"
}
```

**What happens internally:**

1. The document tree is loaded (from local cache or PageIndex).
2. The LLM receives the tree (titles + summaries only — no full text) and the question.
3. The LLM returns a list of `node_id` values that likely contain the answer.
4. The full text of those nodes is assembled as context.
5. The LLM produces a final answer using only that context.

**Response:**

```json
{
  "answer": "The paper concludes that...",
  "thinking": "The question asks about conclusions, which are typically in the final section...",
  "selected_nodes": [
    {
      "id": "node_42",
      "title": "Conclusion",
      "summary": "Summary of findings...",
      "text_preview": "First 300 chars of node text..."
    }
  ],
  "doc_id": "pi-xxxxxxxxxxxx"
}
```

---

### `GET /documents`

List all documents — merges the local cache with the remote PageIndex document list.

**Response:**

```json
{
  "documents": [ { "id": "...", "name": "...", "status": "completed", "cached_filename": "paper.pdf" } ],
  "cache": { "paper.pdf": "pi-xxxxxxxxxxxx" }
}
```

---

### `GET /pdf/{filename}`

Serve a previously uploaded PDF for in-browser preview.

---

## Avoiding Re-indexing

Indexing a document with PageIndex takes time and consumes API quota. The system uses a **three-layer lookup** to ensure a PDF is never re-indexed unnecessarily:

### Layer 1 — Local cache (`doc_id_cache.json`)

On every upload, the mapping `filename → doc_id` is persisted in `doc_id_cache.json` next to the API. On the next upload of the same filename, the system:

1. Reads the cached `doc_id`.
2. Calls `GET /document/{doc_id}` on PageIndex to confirm it still exists and has `status: completed`.
3. If confirmed, returns the cached ID immediately — **zero re-indexing, one API call**.

### Layer 2 — Remote document list scan

If the local cache misses (e.g. cache file deleted, or first run on a new machine), the system pages through your PageIndex account's document list looking for a document whose `name` matches the uploaded filename and whose `status` is `completed`.

If found, the `doc_id` is written to the local cache for future lookups.

### Layer 3 — Fresh submission

Only if both layers miss does the system submit the PDF to PageIndex as a new document.

```
Upload PDF
    │
    ▼
Local cache hit? ──yes──► Verify still completed ──yes──► Use cached doc_id ✅
    │ no                         │ no (deleted/failed)
    │                       Remove stale cache entry
    ▼
Remote list scan ──found──► Save to cache → Use doc_id ✅
    │ not found
    ▼
Submit new document → Save to cache → Wait for processing ✅
```

### Tree caching

After the first `/tree/{doc_id}` or `/query` call, the tree JSON is written to `output/<doc_id>/tree.json`. Subsequent calls load it from disk without touching the PageIndex API.

To force a fresh tree fetch (e.g. after a document re-index), delete the corresponding `output/<doc_id>/tree.json` file.

---

## Frontend

The Streamlit UI (`frontend/app.py`) connects to the backend at `http://127.0.0.1:8001` and provides:

- **PDF upload** with automatic indexing status polling.
- **Chat interface** — ask questions against any indexed document.
- **Tree explorer** — visualise the hierarchical document structure.
- **Document list** — switch between previously indexed documents without re-uploading.

Run it with:

```bash
cd frontend
streamlit run app.py
```
