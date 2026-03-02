"""
FastAPI Application — Enhanced RAG Cache API
=============================================

Endpoints:
  POST /ingest            — Ingest a document with chosen chunking strategy
  POST /chat              — Query the RAG pipeline (three-tier cache aware)
  GET  /cache/stats       — Cache analytics for all three tiers
  DELETE /cache/clear     — Wipe all cache entries
  GET  /health            — Health check (Redis + Pinecone connectivity)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

load_dotenv()

from src.utils.logger import setup_logging, get_logger
setup_logging()

from src.caching import get_cache_backend
from src.caching import parent_cache
from src.ingestion import ingest_document, ChunkStrategy
from src.pipeline import run_query

logger = get_logger(__name__)

app = FastAPI(
    title="Enhanced RAG Cache API",
    description=(
        "Production-grade RAG with Parent-Child & Structure-Recursive chunking, "
        "automatic PDF image enrichment (Groq vision) and table-aware chunking, "
        "and a three-tier (Exact / Semantic / Retrieval) caching system."
    ),
    version="2.0.0",
    docs_url=None,   # served below via custom route (unpkg CDN — no jsdelivr)
    redoc_url=None,
)

# Swagger UI and ReDoc served from unpkg (more reliable than jsdelivr)
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

@app.get("/docs", include_in_schema=False)
def custom_swagger():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Enhanced RAG Cache API",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
def custom_redoc():
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="Enhanced RAG Cache API",
        redoc_js_url="https://unpkg.com/redoc@2/bundles/redoc.standalone.js",
    )

# ── Pydantic Models ───────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    filepath: str = Field(..., description="Absolute server-side path to a .pdf, .txt, or .md file")
    strategy: Optional[str] = Field(
        None,
        description="'parent_child' | 'structure_recursive' "
                    "(default: config.yaml ingestion.default_strategy). "
                    "PDF files are always enriched with Groq vision and table-aware chunking.",
    )
    namespace: Optional[str] = Field(None, description="Pinecone namespace (optional)")


class IngestResponse(BaseModel):
    doc_id: str
    source: str
    strategy: str
    parent_count: int
    chunk_count: int
    upserted: int
    message: str
    breakdown: Optional[Dict[str, Any]] = None  # populated by pdf_rich strategy


class SourceItem(BaseModel):
    citation: str
    source: str
    section: str
    doc_id: str
    score: float
    chunk_preview: str


class ChatRequest(BaseModel):
    query: str = Field(..., description="User question")
    use_reranker: bool = Field(True, description="Enable Pinecone BGE reranking")
    debug: bool = Field(False, description="Include raw chunks and pipeline steps in response")
    source: Optional[str] = Field(None, description="Optional document source filter (scopes cache lookups)")


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]
    cache_hit: bool
    cache_tier: Optional[int]
    cache_tier_name: Optional[str]
    cache_similarity: Optional[float]
    tier3_regen: bool
    total_latency_ms: float
    pipeline_steps: Optional[List[str]] = None
    raw_chunks: Optional[List[Dict[str, Any]]] = None


class CacheClearResponse(BaseModel):
    deleted: Dict[str, int]
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """Ingest a document from a server-side file path."""
    try:
        result = ingest_document(
            file_path=req.filepath,
            strategy=req.strategy,         # type: ignore[arg-type]
            namespace=req.namespace,
        )
        # Bump doc version so stale cache entries are invalidated
        get_cache_backend().bump_doc_version()
        logger.info(f"Doc version bumped after ingesting: {req.filepath}")

        return IngestResponse(
            **result,
            message=f"Ingestion successful using '{req.strategy or 'default'}' strategy.",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Ingestion error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    strategy: str = Form("parent_child"),
    namespace: Optional[str] = Form(None),
):
    """Upload and ingest a document file directly.

    PDF files are automatically enriched with Groq vision image descriptions
    and table-aware placeholder chunking regardless of the chosen strategy.
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    dest = data_dir / file.filename

    # Read file bytes first for hash deduplication
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File read error: {e}")

    file_hash = hashlib.sha256(content).hexdigest()
    cache = get_cache_backend()
    existing = cache.get_document_hash(file_hash)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Duplicate: this file was already uploaded as "
                f"'{existing['file_name']}'"
            ),
        )

    try:
        dest.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save error: {e}")

    try:
        result = ingest_document(
            file_path=str(dest),
            strategy=strategy,              # type: ignore[arg-type]
            namespace=namespace,
        )

        # Store document hash to prevent future duplicate uploads
        cache.set_document_hash(file_hash, {
            "file_name":   file.filename,
            "file_size":   len(content),
            "chunk_count": result.get("chunk_count", 0),
        })

        # Bump doc version so stale cache entries are invalidated
        cache.bump_doc_version()
        logger.info(f"Doc version bumped after upload: {file.filename}")

        return IngestResponse(
            **result,
            message=f"Ingestion successful using '{strategy}' strategy.",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Upload ingestion error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Run the RAG query pipeline with three-tier caching.

    Returns the answer plus rich cache metadata so the frontend can display
    which cache tier was hit (or that it was a full pipeline run).
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        result = run_query(
            query=req.query,
            use_reranker=req.use_reranker,
            debug=req.debug,
            source_filter=req.source or "",
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.exception("Chat pipeline error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats")
def cache_stats():
    """Return hit/entry counts for all three cache tiers plus the parent store."""
    raw = get_cache_backend().get_stats()
    raw["parent_cache"] = parent_cache.stats()
    return raw


@app.delete("/cache/clear", response_model=CacheClearResponse)
def cache_clear():
    """Wipe all three cache tiers and the parent chunk store."""
    cache = get_cache_backend()
    deleted = cache.clear_all()
    deleted["parent"] = parent_cache.flush_all()
    return CacheClearResponse(
        deleted=deleted,
        message="All caches cleared successfully.",
    )


@app.post("/cache/cleanup")
def cache_cleanup():
    """Purge expired entries from all cache tiers. Returns count removed."""
    removed = get_cache_backend().cleanup_expired()
    return {"removed": removed, "message": f"Removed {removed} expired cache entries."}


@app.get("/health")
def health():
    """Health check — verifies Redis + Pinecone connectivity."""
    from src.caching.redis_client import get_redis_client
    import os
    redis_ok = get_redis_client() is not None
    pinecone_ok = bool(os.getenv("PINECONE_API_KEY"))
    return {
        "status": "ok" if (redis_ok and pinecone_ok) else "degraded",
        "redis": "connected" if redis_ok else "unavailable",
        "pinecone": "configured" if pinecone_ok else "missing PINECONE_API_KEY",
        "timestamp": time.time(),
    }


@app.get("/")
def root():
    return {
        "service": "Enhanced RAG Cache API",
        "version": "2.0.0",
        "docs": "/docs",
    }
