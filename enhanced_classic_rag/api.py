"""
FastAPI Application — Enhanced RAG API with incremental chunk diffing.

Endpoints:
- POST /ingest: Ingest a document with deduplication and incremental updates
- POST /chat: Chat with documents using RAG
- POST /search: Search documents without generation
- GET /documents: Get user's documents
- GET /health: Health check
"""

import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import tempfile
import os

from app.core.config import settings
from app.pipeline.processing_pipeline import ProcessingPipeline
from app.vectorstore.reranker import Reranker
from app.generation import Generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced RAG API with Incremental Diff",
    description="Document ingestion with chunk-level deduplication, versioning, and incremental updates",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pipeline = ProcessingPipeline()
reranker = Reranker(
    api_key=settings.PINECONE_API_KEY,
    index_name=settings.PINECONE_INDEX_NAME,
    rerank_model=settings.RERANKER_MODEL
)
generator = Generator()


# ── Pydantic Models ──────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    status: str
    message: str
    filename: str
    username: str
    version: Optional[int] = None
    old_version: Optional[int] = None
    new_version: Optional[int] = None
    doc_id: Optional[int] = None
    total_chunks: Optional[int] = None
    text_chunks: Optional[int] = None
    table_chunks: Optional[int] = None
    chunks_added: Optional[int] = None
    chunks_deleted: Optional[int] = None
    unchanged_chunks: Optional[int] = None
    diff_report: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    username: str = Field(..., description="Username for namespace isolation")
    top_k: int = Field(default=10, description="Number of results to return")


class SearchResult(BaseModel):
    id: str
    score: float
    chunk_text: str
    source: str
    page_number: int
    content_type: str
    version: int


class SearchResponse(BaseModel):
    query: str
    username: str
    results: List[SearchResult]
    total: int


class ChatRequest(BaseModel):
    question: str = Field(..., description="User question")
    username: str = Field(..., description="Username for namespace isolation")
    use_reranker: bool = Field(default=True, description="Whether to use reranking")
    top_k: int = Field(default=10, description="Number of candidates to retrieve")
    top_n: int = Field(default=5, description="Number of results after reranking")


class Citation(BaseModel):
    number: int
    source: str
    page: int
    version: int
    content_type: str
    score: float
    text_preview: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    username: str
    citations: List[Citation]
    reranked: bool


class DocumentRecord(BaseModel):
    id: int
    filename: str
    username: str
    document_hash: str
    version: int
    status: str
    is_active: bool
    uploaded_at: str


class DocumentsResponse(BaseModel):
    username: str
    documents: List[DocumentRecord]
    total: int


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Enhanced RAG API",
        "version": "1.0.0",
        "description": "Document ingestion and retrieval with deduplication and versioning",
        "endpoints": {
            "ingest": "POST /ingest",
            "chat": "POST /chat",
            "search": "POST /search",
            "documents": "GET /documents/{username}",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Pinecone connection
        stats = pipeline.pinecone_manager.get_stats()

        # Ensure stats is JSON-serializable (convert non-serializable parts to strings)
        import json

        def safe_serialize(obj):
            try:
                return json.loads(json.dumps(obj))
            except Exception:
                return str(obj)

        try:
            index_stats = json.loads(json.dumps(stats, default=lambda o: str(o)))
        except Exception:
            index_stats = safe_serialize(stats)

        return {
            "status": "healthy",
            "database": "connected",
            "pinecone": "connected",
            "index_stats": index_stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(..., description="PDF document to ingest"),
    username: str = Form(..., description="Username for namespace isolation")
):
    """
    Ingest a document with automatic deduplication, versioning, and incremental updates.
    
    - Computes SHA256 hash for document-level deduplication
    - Computes SHA256 hash for each chunk
    - Assigns version numbers automatically
    - For updates: performs incremental diff
      - Adds only new chunks
      - Deletes removed chunks
      - Skips unchanged chunks (no re-embedding)
    - Extracts text and tables separately
    - Chunks with different strategies for text and tables
    - Upserts to user-specific namespace in Pinecone
    """
    logger.info(f"Ingestion request: {file.filename} from user '{username}'")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Process document through pipeline (with incremental diff support)
        # Pass the original uploaded filename so metadata `source` is correct
        result = pipeline.process_document(temp_path, username, original_filename=file.filename)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return IngestResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ingestion: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search documents without answer generation.
    
    Returns relevant chunks from user's namespace.
    """
    logger.info(f"Search request: '{request.query}' from user '{request.username}'")
    
    try:
        results = pipeline.search_documents(
            query=request.query,
            username=request.username,
            top_k=request.top_k
        )
        
        search_results = [
            SearchResult(
                id=r["id"],
                score=r["score"],
                chunk_text=r["chunk_text"],
                source=r["source"],
                page_number=r["page_number"],
                content_type=r["content_type"],
                version=r["version"]
            )
            for r in results
        ]
        
        return SearchResponse(
            query=request.query,
            username=request.username,
            results=search_results,
            total=len(search_results)
        )
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with documents using RAG.
    
    - Retrieves relevant chunks from user's namespace
    - Optionally reranks results for better relevance
    - Generates answer using LLM with citations
    """
    logger.info(
        f"Chat request: '{request.question}' from user '{request.username}' "
        f"(rerank: {request.use_reranker})"
    )
    
    try:
        # Retrieve or rerank
        if request.use_reranker:
            chunks = reranker.rerank(
                query=request.question,
                namespace=request.username.lower(),
                top_k=request.top_k,
                top_n=request.top_n
            )
        else:
            chunks = pipeline.search_documents(
                query=request.question,
                username=request.username,
                top_k=request.top_k
            )
        
        if not chunks:
            return ChatResponse(
                question=request.question,
                answer="I couldn't find any relevant information in your documents.",
                username=request.username,
                citations=[],
                reranked=request.use_reranker
            )
        
        # Generate answer
        result = generator.generate_with_citations(request.question, chunks)
        
        # Format citations
        citations = [
            Citation(
                number=c["number"],
                source=c["source"],
                page=c["page"],
                version=c["version"],
                content_type=c["content_type"],
                score=c["score"],
                text_preview=c["text_preview"]
            )
            for c in result["citations"]
        ]
        
        return ChatResponse(
            question=request.question,
            answer=result["answer"],
            username=request.username,
            citations=citations,
            reranked=request.use_reranker
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{username}", response_model=DocumentsResponse)
async def get_user_documents(username: str, limit: int = 50):
    """
    Get all documents for a user.
    
    Returns document metadata including versions, status, and upload dates.
    """
    logger.info(f"Documents request for user '{username}'")
    
    try:
        documents = pipeline.get_user_documents(username, limit)
        
        doc_records = [
            DocumentRecord(
                id=doc["id"],
                filename=doc["filename"],
                username=doc["username"],
                document_hash=doc["document_hash"],
                version=doc["version"],
                status=doc["status"],
                is_active=doc["is_active"],
                uploaded_at=doc["uploaded_at"]
            )
            for doc in documents
        ]
        
        return DocumentsResponse(
            username=username,
            documents=doc_records,
            total=len(doc_records)
        )
        
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🚀 Starting Enhanced RAG API Server")
    print("=" * 60)
    print(f"Index: {settings.PINECONE_INDEX_NAME}")
    print(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"Reranker Model: {settings.RERANKER_MODEL}")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
