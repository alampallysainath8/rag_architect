# ⚡ Enhanced RAG Cache

> Production-grade RAG · Parent-Child chunking · 3-Tier intelligent caching · Pinecone + Redis + OpenAI

---

## Flow

```mermaid
flowchart TD
        A([User Query]) --> B[Normalize + Hash]

        B --> C{Tier 1\nExact Cache}
        C -- HIT --> Z1([Return Answer])
        C -- MISS --> D[Embed Query\nopenai text-embedding-3-small]

        D --> E{Tier 2\nSemantic Cache\ncosine ≥ 0.92}
        E -- HIT --> Z2([Return Answer])
        E -- MISS --> F{Tier 3\nRetrieval Cache\ncosine ≥ 0.80}

        F -- HIT --> G[Re-inject Parent\nfrom Redis]
        G --> H[LLM\nGPT-4o-mini]
        H --> Z3([Return Answer])

        F -- MISS --> I[Pinecone\nVector Search]
        I --> J[BGE Reranker]
        J --> K[group_children_by_parent\nDeduplicate parent IDs]
        K --> L[fetch_parent_chunks\nRedis — 1 GET per unique parent]
        L --> M[merge_children_to_parents\nUnique parent texts only]
        M --> H

        H --> N[(Write-through\nAll 3 Tiers)]

        %% Ingestion subgraph: PDF -> enriched markdown -> chunking -> upsert
        subgraph ING["Ingestion (offline) — PDF → Enriched Markdown → Chunking"]
            direction TB
            P0[PDF -> Markdown\npymupdf4llm] --> P1[Extract image URLs\nand alt-text placeholders]
            P1 --> P2[Image Enricher\nGroq vision -> image summaries]
            P2 --> P6["Enriched Markdown\nmarkdown and image summaries"]
            P6 --> P3[Structure-aware chunking\nstructure_recursive]
            P6 --> P4[Parent/Child chunking\nparent_child]
            P3 --> P5[Embed & Upsert\ninto Pinecone]
            P4 --> P5
        end

        P5 --> I

        style Z1 fill:#d1fae5,stroke:#10b981,color:#065f46
        style Z2 fill:#ccfbf1,stroke:#14b8a6,color:#0f766e
        style Z3 fill:#e0f2fe,stroke:#0ea5e9,color:#075985
        style N  fill:#fef9c3,stroke:#eab308,color:#713f12
```

---

## Project Structure

```
enhanced_rag_cache/
├── api.py                      # FastAPI routes (ingest, chat, /documents, stats)
├── main.py                     # Uvicorn entry point
├── config.yaml                 # All tunable parameters
├── docker-compose.yml          # Redis
├── src/
│   ├── pipeline.py             # 3-tier cache → retrieval → LLM orchestration
│   ├── ingestion.py            # PDF enrichment + chunking + Pinecone upsert
│   ├── chunking/
│   │   ├── parent_child.py         # Large parents (Redis) + small children (Pinecone)
│   │   └── structure_recursive.py  # Header-aware + recursive fallback
│   ├── caching/
│   │   ├── cache_manager.py        # Tier 1 / 2 / 3 orchestrator
│   │   ├── redis_backend.py        # Redis implementation
│   │   ├── sqlite_backend.py       # SQLite fallback
│   │   ├── parent_cache.py         # Parent chunk store (Redis)
│   │   └── redis_client.py
│   ├── retrieval/
│   │   ├── retriever.py            # Pinecone vector search
│   │   ├── reranker.py             # BGE reranking + parent merge
│   │   ├── parent_merger.py        # group → deduplicate → fetch parents
│   │   └── pinecone_manager.py
│   ├── generation/
│   │   └── generator.py            # GPT-4o-mini answer synthesis
│   └── utils/
│       ├── embeddings.py           # OpenAI text-embedding-3-small
│       ├── pdf_to_markdown.py      # pymupdf4llm
│       ├── image_enricher.py       # Groq vision image descriptions
│       └── config_loader.py
├── frontend/
│   └── app.py                  # Streamlit UI
├── screenshots/
│   ├── rag_cache.png
│   ├── rag_cache_redis.png
│   ├── rag_cache_swagger.png
│   └── demo_rag_cache.mp4
└── data/                       # Drop documents here
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env   # fill PINECONE_API_KEY + OPENAI_API_KEY

# 3. Redis
docker-compose up -d

# 4. API
python main.py          # → http://localhost:8000/docs

# 5. UI (separate terminal)
cd frontend && streamlit run app.py   # → http://localhost:8501
```

---

## Stack

| | |
|---|---|
| Vector DB | Pinecone (integrated embedding) |
| Cache | Redis 7 · 3-tier (Exact / Semantic / Retrieval) |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Reranker | Pinecone BGE-reranker-v2-m3 |
| PDF | pymupdf4llm + Groq vision |
| API | FastAPI + Uvicorn |
| UI | Streamlit |

---

## API

| Method | Endpoint | |
|---|---|---|
| `POST` | `/ingest` | Ingest by file path |
| `POST` | `/ingest/upload` | Upload + ingest |
| `GET` | `/documents` | List ingested documents |
| `POST` | `/chat` | Query (3-tier cache aware) |
| `GET` | `/cache/stats` | Cache analytics |
| `DELETE` | `/cache/clear` | Wipe all caches |
| `GET` | `/health` | Health check |

---

## Screenshots

### UI
![UI](screenshots/rag_cache.png)

### Redis Cache
![Redis](screenshots/rag_cache_redis.png)

### Swagger
![Swagger](screenshots/rag_cache_swagger.png)

### Demo
> 📹 [`demo_rag_cache.mp4`](screenshots/demo_rag_cache.mp4)
