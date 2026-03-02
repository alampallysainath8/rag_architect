# Enhanced RAG Cache — Implementation Guide

> Explains the full data flow, every input/output at each layer, setup
> instructions, and how to run the tests.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Ingestion Flow](#2-ingestion-flow)
   - [Strategy A — Parent-Child](#strategy-a--parent-child)
   - [Strategy B — Structure-Recursive](#strategy-b--structure-recursive)
   - [Strategy C — PDF Rich (images + tables)](#strategy-c--pdf-rich-images--tables)
3. [Query Flow (Three-Tier Cache)](#3-query-flow-three-tier-cache)
4. [Caching Layer Deep Dive](#4-caching-layer-deep-dive)
5. [Inputs Required](#5-inputs-required)
6. [How to Run](#6-how-to-run)
7. [How to Test](#7-how-to-test)
8. [File → Module Map](#8-file--module-map)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION SIDE                               │
│                                                                     │
│  PDF / TXT / MD                                                     │
│       │                                                             │
│       ▼                                                             │
│  pdf_to_markdown.py  ──(embed_images=True for pdf_rich)──▶         │
│  image_enricher.py   ──(Groq vision → description+table+JSON)──▶   │
│       │                                                             │
│       ▼  (choose strategy)                                          │
│  ┌──────────────┐  ┌───────────────────┐  ┌──────────────────────┐ │
│  │ parent_child │  │structure_recursive│  │   table_aware        │ │
│  │  chunker     │  │     chunker       │  │   chunker            │ │
│  └──────┬───────┘  └────────┬──────────┘  └──────────┬───────────┘ │
│         │                  │                          │             │
│         ▼                  ▼                          ▼             │
│  Redis (parents)     Pinecone upsert           Pinecone upsert     │
│  Pinecone (children)                                                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         QUERY SIDE                                  │
│                                                                     │
│  User Query                                                         │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────────────────────────────┐                          │
│  │      THREE-TIER CACHE LOOKUP         │                          │
│  │                                      │                          │
│  │  Tier 1: Exact   (SHA-256 key match) │──HIT──▶ return answer   │
│  │  Tier 2: Semantic(cos-sim ≥ 0.92)   │──HIT──▶ return answer   │
│  │  Tier 3: Retrieval(cos-sim ≥ 0.80)  │──HIT──▶ skip Pinecone  │
│  └──────────────────────────────────────┘         ▼               │
│                    │ MISS                    LLM generation        │
│                    ▼                                               │
│  Pinecone search (top_k=10)                                        │
│       │                                                             │
│       ▼                                                             │
│  BGE Reranker (top_n=5) + Redis parent injection                   │
│       │                                                             │
│       ▼                                                             │
│  GPT-4o-mini generation (cited answer)                             │
│       │                                                             │
│       ▼                                                             │
│  Write-through → all 3 Redis tiers                                 │
│       │                                                             │
│       ▼                                                             │
│  Response { answer, sources, cache_tier, latency_ms }              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Ingestion Flow

Entry point: `src/ingestion.py → ingest_document(file_path, strategy, namespace)`

### Strategy A — Parent-Child

Best for: general documents, dense narrative text.

```
PDF / TXT / MD
     │
     ▼
file_to_text()                      → plain Markdown (no images)
     │
     ▼
parent_child_chunk()
     │
     ├─► ParentChunk list  ──store_parents()──► Redis
     │     key: rag_cache:parent:<parent_id>
     │     TTL: 7 days
     │     payload: { parent_id, doc_id, source, text, index }
     │
     └─► ChildChunk list  ──upsert_records()──► Pinecone
           Pinecone record per child:
           {
             _id:           "<doc_id>::parent_<n>::child_<m>",
             chunk_text:    "<child text>" ,          ← indexed + embedded
             source:        "<filename>",
             doc_id:        "<filename>",
             parent_id:     "<doc_id>::parent_<n>",   ← Redis lookup key
             level:         "child",
             chunk_index:   "<int>",
           }
```

**Config knobs** (`config.yaml → chunking.parent_child`):

| Setting | Default | Meaning |
|---|---|---|
| `parent_chunk_size` | 1500 | Max chars per parent |
| `parent_overlap` | 100 | Overlap between parents |
| `child_chunk_size` | 300 | Max chars per child |
| `child_overlap` | 50 | Overlap between children |

---

### Strategy B — Structure-Recursive

Best for: well-structured documents (PDFs with H1–H4 headings).

```
PDF / TXT / MD
     │
     ▼
file_to_text()                      → Markdown (pymupdf4llm heading-aware)
     │
     ▼
structure_recursive_chunk()
     │
     ├─► MarkdownHeaderTextSplitter  (split on #, ##, ###, ####)
     │       each split = one section with heading metadata
     │
     └─► If section > max_section_size (1200 chars):
             RecursiveCharacterTextSplitter fallback
             sub-chunks carry same heading metadata, sub_index > 0
     │
     ▼
upsert_records() ──► Pinecone
  Pinecone record per StructChunk:
  {
    _id:         "<doc_id>::struct_<n>[_sub_<m>]",
    chunk_text:  "<text>",
    source:      "<filename>",
    doc_id:      "<filename>",
    strategy:    "structure_recursive",
    h1:          "Financial Overview",   ← from Markdown headers
    h2:          "Revenue Breakdown",
    chunk_index: "<int>",
    sub_index:   "<int>",               ← 0 unless recursively split
  }
```

**Config knobs** (`config.yaml → chunking.structure_recursive`):

| Setting | Default | Meaning |
|---|---|---|
| `max_section_size` | 1200 | Chars before recursive split kicks in |
| `recursive_chunk_size` | 600 | Target chunk size for sub-splits |
| `recursive_overlap` | 80 | Overlap for sub-splits |

---

### Strategy C — PDF Rich (images + tables)

Best for: financial reports, presentations, documents with charts and tables.

```
PDF file
  │
  ▼
Step 1 — pdf_to_markdown(embed_images=True)
         pymupdf4llm extracts:
           • text with heading structure
           • tables as Markdown pipes
           • images as inline base64 blobs  ![](data:image/png;base64,...)
  │
  ▼
Step 2 — enrich_markdown_images()
         For each base64 image:
           a. _get_nearby_context()  → surrounding headers / captions / legends (10 lines before, 6 after)
           b. _call_vision_model()   → Groq llama-3.2-90b-vision-preview
              Input to model:
                [text]  Context (section headers, captions, legends): ...
                [text]  Vision prompt (asks for DESCRIPTION / TABLE / JSON sections)
                [image] base64 PNG
              Model output (3 labelled sections):
                ---DESCRIPTION---  prose paragraph about the chart
                ---TABLE---        Markdown table of extracted data
                ---JSON---         { "columns":[...], "rows":[...], "values":[[...]] }
           c. Replace base64 blob with:
                **[Image N — Description]:**    <desc>
                **[Image N — Extracted Table]:** <markdown table>
                **[Image N — JSON Data]:**       ```json { ... } ```
  │
  ▼
Step 3 — table_aware_chunk()
         Pre-processing:
           • Strip page-number-only heading lines (e.g. #### **3**)
           • Extract image blocks → __IMG_BLOCK_N__ tokens
           • Extract Markdown tables + their preceding headers → __TABLE_N__ tokens

         Header splitting on remaining text (MarkdownHeaderTextSplitter)
         → each regular section → RecursiveCharacterTextSplitter if > 1500 chars

         Token reinsertion:
           __TABLE_N__ → chunk_type="table"   (one complete table = one chunk)
           __IMG_BLOCK_N__ → chunk_type="image" (description+table+JSON = one chunk)
           plain text → chunk_type="text"
  │
  ▼
upsert_records() ──► Pinecone
  Pinecone record per TableAwareChunk:
  {
    _id:            "<doc_id>::tbl_<n>  |  img_<n>  |  txt_<n>",
    chunk_text:     "<content>",
    source:         "<filename>",
    doc_id:         "<filename>",
    chunk_type:     "table" | "image" | "text",
    strategy:       "table_aware",
    chunk_index:    "<int>",
    # tables also carry:
    table_rows:     "<int>",
    table_columns:  "<int>",
    # image chunks also carry:
    image_number:   "<int>",
    # text chunks also carry heading metadata:
    h1..h4:         "<heading text>",
  }
```

---

## 3. Query Flow (Three-Tier Cache)

Entry point: `src/pipeline.py → run_query(query, use_reranker, debug)`

```
query (str)
  │
  ▼
CacheManager.lookup(query)
  │
  ├─ Tier 1 — Exact Cache
  │    Key: rag_cache:exact:<SHA-256 of normalised query>
  │    Normalisation: lowercase + strip punctuation + collapse whitespace
  │    HIT → return { answer, sources }  (no Pinecone, no LLM)
  │
  ├─ Tier 2 — Semantic Cache
  │    Embed query with text-embedding-3-small (OpenAI)
  │    Compare cosine similarity against all cached embeddings
  │    Threshold: 0.92
  │    HIT → return { answer, sources }  (no Pinecone, no LLM)
  │
  └─ Tier 3 — Retrieval Cache
       Same embedding, lower threshold: 0.80
       HIT → cached_chunks returned (skip Pinecone)
              ↓
              rerank_preloaded() — re-inject parent context from Redis
              ↓
              generate_answer()  — LLM called with stale chunks

MISS (or Tier-3 hit falls through to generation):
  │
  ▼
Pinecone search  (top_k = 10)
  │
  ▼  (if use_reranker=True)
Pinecone BGE rerank (bge-reranker-v2-m3, top_n = 5)
  + parent context injection:
      for each child chunk where parent_id is set:
          get_parent(parent_id) from Redis
          set context_text = full parent text
  │
  ▼
generate_answer(query, chunks)
  OpenAI GPT-4o-mini
  System prompt includes numbered context blocks, instructs citation [n]
  Returns answer string with inline citations
  │
  ▼
format_sources(chunks)   → deduplicated source list
  │
  ▼
CacheManager.store(query, answer, sources, chunks)
  ├─ exact_cache.set()      → Redis key with 24h TTL
  ├─ semantic_cache.set()   → Redis embedding + payload, 12h TTL
  └─ retrieval_cache.set()  → Redis embedding + chunk list, 6h TTL
  │
  ▼
Response:
  {
    query, answer, sources,
    cache_hit, cache_tier, cache_tier_name, cache_similarity,
    tier3_regen, total_latency_ms,
    pipeline_steps (if debug=True),
    raw_chunks     (if debug=True),
  }
```

---

## 4. Caching Layer Deep Dive

### Redis Key Schema

```
rag_cache:exact:<sha256>                    → JSON { answer, sources, cached_at }
rag_cache:semantic:index                    → Hash { entry_id → JSON meta }
rag_cache:semantic:emb:<entry_id>           → JSON embedding vector [float × 1536]
rag_cache:semantic:payload:<entry_id>       → JSON { answer, sources }
rag_cache:retrieval:index                   → Hash { entry_id → JSON meta }
rag_cache:retrieval:emb:<entry_id>          → JSON embedding vector
rag_cache:retrieval:chunks:<entry_id>       → JSON list of chunk dicts
rag_cache:parent:<parent_id>                → JSON { parent_id, doc_id, source, text }
```

### TTL Summary

| Tier | TTL | Rationale |
|---|---|---|
| Tier 1 Exact | 24 h | Likely to be re-asked same day |
| Tier 2 Semantic | 12 h | Paraphrases re-asked within hours |
| Tier 3 Retrieval | 6 h | Chunks may drift as new docs ingested |
| Parent chunks | 7 days | Static for the lifetime of a document |

### Embedding Model

OpenAI `text-embedding-3-small` (1536-dim) is used **only** for the Redis cache tiers.  
Pinecone uses its own server-side embedding (`multilingual-e5-large`) for upsert and search — no local embedding needed for retrieval.

---

## 5. Inputs Required

### Environment variables (`.env`)

```dotenv
# Required
PINECONE_API_KEY=pc-...         # Pinecone API key
OPENAI_API_KEY=sk-...           # OpenAI API key (GPT-4o-mini + embeddings)
GROQ_API_KEY=gsk_...            # Groq API key (only for pdf_rich strategy)

# Optional overrides
REDIS_HOST=localhost             # default: localhost
REDIS_PORT=6379                  # default: 6379
REDIS_DB=0                       # default: 0
REDIS_PASSWORD=                  # leave blank if no auth
```

### config.yaml (key sections)

```yaml
pinecone:
  index_name: "rag-cache-v2"     # will be created automatically if absent
  cloud: "aws"
  region: "us-east-1"
  embed_model: "multilingual-e5-large"
  rerank_model: "bge-reranker-v2-m3"
  namespace: "documents"

openai:
  llm_model: "gpt-4o-mini"
  embedding_model: "text-embedding-3-small"

retrieval:
  top_k: 10                      # Pinecone candidates
  rerank_top_n: 5                # kept after reranking
```

### Ingest API payload

`POST /ingest`

```json
{
  "filepath": "/absolute/path/to/report.pdf",
  "strategy": "parent_child",
  "namespace": "financials"
}
```

| Field | Required | Values |
|---|---|---|
| `filepath` | Yes | Absolute server-side file path (.pdf / .txt / .md) |
| `strategy` | No (default: `parent_child`) | `"parent_child"` \| `"structure_recursive"` \| `"pdf_rich"` |
| `namespace` | No | Any string; groups vectors in Pinecone |

### Chat API payload

`POST /chat`

```json
{
  "query": "What was the revenue in Q4 2024?",
  "use_reranker": true,
  "debug": false
}
```

| Field | Required | Meaning |
|---|---|---|
| `query` | Yes | User question |
| `use_reranker` | No (default: true) | Run BGE reranker after Pinecone search |
| `debug` | No (default: false) | Include `pipeline_steps` and `raw_chunks` in response |

---

## 6. How to Run

### Prerequisites

- Python 3.10+
- Redis running locally (or via Docker)
- Pinecone account (Serverless, AWS us-east-1)
- OpenAI API key
- Groq API key (only needed for `pdf_rich`)

### Redis (Docker — quickest)

```bash
docker run -d --name redis-rag -p 6379:6379 redis:7
```

### Install

```bash
cd enhanced_rag_cache

# Create and activate virtualenv
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

### Environment

```bash
copy .env.example .env          # Windows
# cp .env.example .env          # Linux/macOS
# fill in your keys in .env
```

### Start the API

```bash
python main.py
# Starts uvicorn on http://127.0.0.1:8000
# Interactive docs: http://127.0.0.1:8000/docs
```

### Start the Streamlit UI

```bash
cd frontend
streamlit run app.py
# Opens http://localhost:8501
```

### Ingest a document (curl)

```bash
# parent_child strategy
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"filepath": "C:/data/report.pdf", "strategy": "parent_child"}'

# pdf_rich strategy (images + tables — requires GROQ_API_KEY)
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"filepath": "C:/data/report.pdf", "strategy": "pdf_rich"}'
```

### Ingest via file upload

```bash
curl -X POST http://127.0.0.1:8000/ingest/upload \
  -F "file=@report.pdf" \
  -F "strategy=structure_recursive"
```

### Run a query (curl)

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the revenue in Q4 2024?", "use_reranker": true}'
```

### Check health

```bash
curl http://127.0.0.1:8000/health
# { "status": "ok", "redis": "connected", "pinecone": "configured" }
```

### Clear cache

```bash
curl -X DELETE http://127.0.0.1:8000/cache/clear
```

---

## 7. How to Test

Tests are in `tests/`. Each file covers one layer with **2 test cases**.  
All tests mock external services (Redis, Pinecone, OpenAI, Groq) — no live credentials needed.

### Install test dependencies

```bash
pip install pytest pytest-mock
```

### Run all tests

```bash
cd enhanced_rag_cache
pytest tests/ -v
```

### Run one layer

```bash
pytest tests/test_chunking.py   -v    # chunking layer
pytest tests/test_caching.py    -v    # Redis cache tiers
pytest tests/test_retrieval.py  -v    # Pinecone search + reranker
pytest tests/test_ingestion.py  -v    # ingestion pipeline
pytest tests/test_generation.py -v    # LLM generation + source formatting
```

### What each test verifies

| File | Case 1 | Case 2 |
|---|---|---|
| `test_chunking.py` | Parent IDs, child→parent cross-reference, all Pinecone fields present | Heading metadata in `h1`/`h2` keys; oversized sections produce `sub_index > 0` |
| `test_caching.py` | Exact cache set→get round-trip + query normalisation; different query returns None | Retrieval cache chunk list stored in Redis and recovered with cosine-sim ≥ threshold |
| `test_retrieval.py` | `rerank_preloaded` sets `context_text` to full parent text from Redis | `search()` returns dicts with all required keys (`id`, `score`, `chunk_text`, `h1`…) |
| `test_ingestion.py` | `FileNotFoundError` raised when file doesn't exist | `store_parents` and `upsert_records` each called once; summary dict has correct shape |
| `test_generation.py` | `format_sources` deduplicates same source+section across multiple chunks | `generate_answer([])` returns hardcoded fallback without calling OpenAI |

### Expected output (all passing)

```
tests/test_chunking.py::test_parent_child_chunk_ids_and_metadata       PASSED
tests/test_chunking.py::test_structure_recursive_heading_meta_and_sub_index PASSED
tests/test_caching.py::test_exact_cache_hit_miss_and_normalisation      PASSED
tests/test_caching.py::test_retrieval_cache_stores_and_retrieves_chunks PASSED
tests/test_retrieval.py::test_rerank_preloaded_injects_parent_context   PASSED
tests/test_retrieval.py::test_search_returns_normalised_hit_dicts       PASSED
tests/test_ingestion.py::test_ingest_document_raises_for_missing_file   PASSED
tests/test_ingestion.py::test_ingest_document_parent_child_summary      PASSED
tests/test_generation.py::test_format_sources_deduplication             PASSED
tests/test_generation.py::test_generate_answer_returns_fallback_for_empty_chunks PASSED

10 passed in ~0.3s
```

---

## 8. File → Module Map

```
src/
├── ingestion.py              ingest_document(file_path, strategy, namespace)
├── pipeline.py               run_query(query, use_reranker, debug)
│
├── chunking/
│   ├── parent_child.py       parent_child_chunk(text, doc_id, source, …)
│   │                         → (List[ParentChunk], List[ChildChunk])
│   ├── structure_recursive.py structure_recursive_chunk(md, doc_id, source, …)
│   │                         → List[StructChunk]
│   └── table_aware.py        table_aware_chunk(md, doc_id, source, …)
│                             → List[TableAwareChunk]
│
├── caching/
│   ├── cache_manager.py      CacheManager.lookup(query) / .store(…) / .flush_all()
│   ├── exact_cache.py        get(query) / set(query, payload)
│   ├── semantic_cache.py     get(query) / set(query, payload)
│   ├── retrieval_cache.py    get(query) → List[chunk] / set(query, chunks)
│   ├── parent_cache.py       store_parents(list) / get_parent(parent_id)
│   └── redis_client.py       get_client() / make_key(*parts)
│
├── retrieval/
│   ├── pinecone_manager.py   upsert_records(records) / delete_document(doc_id)
│   ├── retriever.py          search(query, top_k, namespace) → List[hit]
│   └── reranker.py           rerank(query, top_k, top_n) → List[hit]
│                             rerank_preloaded(query, chunks) → List[hit]
│
├── generation/
│   └── generator.py          generate_answer(query, chunks) → str
│                             format_sources(chunks) → List[source_dict]
│
└── utils/
    ├── config_loader.py      cfg  (singleton dict from config.yaml)
    ├── embeddings.py         embed_text(text) / cosine_similarity(a, b)
    ├── pdf_to_markdown.py    file_to_text(path, embed_images=False)
    ├── image_enricher.py     enrich_markdown_images(markdown) → str
    ├── logger.py             get_logger(name) / setup_logging()
    └── exceptions.py         IngestionException / ImageEnrichmentException
                              CacheException / RetrievalException / GenerationException
```
