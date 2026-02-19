from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.ingestion import ingest_document
from app.embedding import upsert_chunks
from app.retrieval import search
from app.rearanker import rerank
from app.generation import generate_answer



app=FastAPI(title="Classic RAG API",description="API for Classic RAG Application")

class IngestRequest(BaseModel):
    filepath:str
    
class IngestResponse(BaseModel):
    file:str
    chunks:int
    message:str
    
class SourceChunk(BaseModel):
    id:str
    score:float
    source:str
    pages:str
    chunk_text:str
    citation:str

class ChatRequest(BaseModel):
    question: str
    use_reranker: bool = True   # toggle reranking on/off
    debug: bool = False  
class ChatResponse(BaseModel):
    answer:str
    sources:List[SourceChunk]
    retrived:Optional[List[SourceChunk]]=None
    reranked:Optional[List[SourceChunk]]=None
    
class GenerateRequest(BaseModel):
    query:str
    top_k:int=5
    top_n:int=5
    use_reranker:bool=True
    
class GenerateResponse(BaseModel):
    question:str
    answer:str
    sources:List[SourceChunk]
    pipeline:str
    
class SearchRequest(BaseModel):
    query:str
    top_k:int=5
    use_reranker:bool=True
    
    
@app.post("/ingest",response_model=IngestResponse)
def ingest(req:IngestRequest):
    """Ingest documents from a specified file path.

    Args:
        req (IngestRequest): _description_
    """
    try:
        records=ingest_document(req.filepath)
        upserted=upsert_chunks(records)
        return IngestResponse(file=req.filepath,chunks=upserted,message="Ingestion successful")
    except FileNotFoundError:
        raise HTTPException(status_code=404,detail=f"File '{req.filepath}' not found")
    except ValueError as ve:
        raise HTTPException(status_code=400,detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    
    
@app.post("/chat",response_model=ChatResponse)
def chat_endpoint(req:ChatRequest):
    """Handle a chat request with optional reranking.

    Args:
        req (ChatRequest): _description_
    """
    try:
        # Step 1: Always fetch raw retrieval results
        retrieved_chunks = search(req.question)

        # Step 2: Rerank if enabled
        if req.use_reranker:
            reranked_chunks = rerank(req.question)
            chunks = reranked_chunks
        else:
            chunks = retrieved_chunks

        if not chunks:
            return ChatResponse(
                answer="I couldn't find any relevant information in the documents.",
                sources=[],
            )
        # Step 3: Generate answer using the selected chunks
        answer = generate_answer(req.question, chunks)
        
        def _to_source(c, idx):
            return SourceChunk(
                id=c["id"],
                score=c["score"],
                source=c["source"],
                pages=c.get("pages", ""),
                chunk_text=c["chunk_text"][:200] + "...",
                citation=f"[{idx}]",
            )
            
        sources=[_to_source(c, idx) for idx, c in enumerate(chunks,1)]
        # Debug: include raw retrieval + reranked lists for comparison
        debug_retrieved = None
        debug_reranked = None
        if req.debug:
            debug_retrieved = [_to_source(c, i) for i, c in enumerate(retrieved_chunks, 1)]
            if req.use_reranker:
                debug_reranked = [_to_source(c, i) for i, c in enumerate(reranked_chunks, 1)]

        return ChatResponse(
            answer=answer,
            sources=sources,
            retrieved=debug_retrieved,
            reranked=debug_reranked,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/search")
def search_endpoint(req: SearchRequest):
    """
    Search only (no generation) — useful for debugging retrieval.
    """
    try:
        if req.use_reranker:
            results = rerank(req.query, top_k=req.top_k)
        else:
            results = search(req.query, top_k=req.top_k)
        return {"query": req.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    """
    Standalone generation endpoint: retrieve → (rerank) → generate.
    Returns the final LLM answer with cited, reranked sources.
    """
    try:
        # Step 1: Retrieve
        retrieved_chunks = search(req.question, top_k=req.top_k)

        # Step 2: Rerank (optional)
        if req.use_reranker:
            chunks = rerank(req.question, top_k=req.top_k, top_n=req.top_n)
            pipeline = "retrieve → rerank → generate"
        else:
            chunks = retrieved_chunks[:req.top_n]
            pipeline = "retrieve → generate"

        if not chunks:
            return GenerateResponse(
                question=req.question,
                answer="No relevant information found in the documents.",
                sources=[],
                pipeline=pipeline,
            )

        # Step 3: Generate
        answer = generate_answer(req.question, chunks)

        sources = [
            SourceChunk(
                id=c["id"],
                score=c["score"],
                source=c["source"],
                pages=c.get("pages", ""),
                chunk_text=c["chunk_text"][:200] + "...",
                citation=f"[{i}]",
            )
            for i, c in enumerate(chunks, 1)
        ]

        return GenerateResponse(
            question=req.question,
            answer=answer,
            sources=sources,
            pipeline=pipeline,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
