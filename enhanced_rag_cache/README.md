# Enhanced RAG Cache

Production-grade RAG pipeline with **intelligent document chunking** and a **three-tier caching system** built on Pinecone, OpenAI, and Redis.

## What's New vs Classic RAG

| Feature | Classic RAG | Enhanced RAG Cache |
|---|---|---|
| Chunking | Fixed-size flat chunks | Parent-Child + Structure-Recursive |
| Caching | None | 3-tier (Exact / Semantic / Retrieval) |
| LLM context | Narrow child chunks | Full parent context sent to LLM |
| Redundant calls | Every query hits LLM | Cached answers skip Pinecone + LLM |

---

## Project Structure

```
enhanced_rag_cache/
├── config.yaml             # All tunable parameters
├── .env.example            # Environment variable template
├── requirements.txt
├── main.py                 # API entry point (uvicorn)
├── api.py                  # FastAPI routes
├── src/
│   ├── ingestion.py        # Document loading + chunking pipeline
│   ├── pipeline.py         # Full query pipeline (cache → retrieval → LLM)
│   ├── chunking/
│   │   ├── parent_child.py          # Strategy 1
│   │   └── structure_recursive.py  # Strategy 2
│   ├── caching/
│   │   ├── cache_manager.py    # Orchestrates all 3 tiers
│   │   ├── exact_cache.py      # Tier 1
│   │   ├── semantic_cache.py   # Tier 2
│   │   ├── retrieval_cache.py  # Tier 3
│   │   ├── parent_cache.py     # Parent chunk store
│   │   └── redis_client.py     # Shared Redis connection
│   ├── retrieval/
│   │   ├── pinecone_manager.py  # Index + upsert
│   │   ├── retriever.py         # Vector search
│   │   └── reranker.py          # BGE reranking + parent injection
│   ├── generation/
│   │   └── generator.py         # OpenAI GPT-4o-mini
│   └── utils/
│       ├── config_loader.py     # config.yaml singleton
│       ├── embeddings.py        # OpenAI text-embedding-3-small
│       ├── pdf_to_markdown.py   # pymupdf4llm PDF→Markdown
│       └── logger.py            # Centralised logging
├── frontend/
│   └── app.py              # Streamlit UI
├── data/                   # Drop your documents here
└── docs/
    └── architecture.md
```

---

## Quick Start

### 1. Clone and install

```bash
cd enhanced_rag_cache
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in:
#   PINECONE_API_KEY=...
#   OPENAI_API_KEY=...
```

### 3. Start Redis

```bash
# Docker (recommended):
docker run -d -p 6379:6379 redis:7

# Or install locally and run: redis-server
```

> **Note:** The app works without Redis — caching is silently disabled and all queries run through the full pipeline.

### 4. Start the API

```bash
python main.py
# API docs: http://localhost:8000/docs
```

### 5. Start the frontend (separate terminal)

```bash
cd frontend
streamlit run app.py
# Opens at http://localhost:8501
```

---

## Configuration

All parameters are in [config.yaml](config.yaml). Key settings:

| Section | Key | Default | Description |
|---|---|---|---|
| `chunking.parent_child` | `parent_chunk_size` | 1500 | Parent chunk size (chars) |
| `chunking.parent_child` | `child_chunk_size` | 300 | Child chunk size (chars) |
| `chunking.structure_recursive` | `max_section_size` | 1200 | Max section before recursive split |
| `cache.semantic` | `similarity_threshold` | 0.92 | Cosine similarity for Tier-2 hit |
| `cache.retrieval` | `similarity_threshold` | 0.80 | Cosine similarity for Tier-3 hit |
| `cache.exact` | `ttl_seconds` | 86400 | Tier-1 TTL (24 h) |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Ingest document by server-side file path |
| `POST` | `/ingest/upload` | Upload + ingest a file directly |
| `POST` | `/chat` | Query with three-tier cache |
| `GET` | `/cache/stats` | Cache analytics |
| `DELETE` | `/cache/clear` | Wipe all caches |
| `GET` | `/health` | Redis + API health check |

### Example: Ingest

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"filepath": "/absolute/path/to/doc.pdf", "strategy": "parent_child"}'
```

### Example: Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings?", "use_reranker": true}'
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Vector DB | Pinecone (integrated embedding) |
| Embeddings (cache) | OpenAI text-embedding-3-small |
| LLM | OpenAI GPT-4o-mini |
| Reranking | Pinecone BGE-reranker-v2-m3 |
| Cache backend | Redis 7 |
| PDF conversion | pymupdf4llm |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full system architecture, chunking strategy comparison, and query flow walkthrough.
