"""
PageIndex RAG – FastAPI Backend
Exposes HTTP endpoints consumed by the Streamlit frontend.
"""

import os
import re
import json
import asyncio
import time
from pathlib import Path

import dotenv
dotenv.load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from pageindex import PageIndexClient
import pageindex.utils as utils
from groq import AsyncGroq

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_FILE = BASE_DIR / "doc_id_cache.json"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────

app = FastAPI(title="PageIndex RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _pi_client() -> PageIndexClient:
    if not PAGEINDEX_API_KEY:
        raise HTTPException(status_code=500, detail="PAGEINDEX_API_KEY not set")
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _extract_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    return match.group(1).strip() if match else text.strip()


async def _call_llm(prompt: str, temperature: float = 0, max_retries: int = 3) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")
    client = AsyncGroq(api_key=GROQ_API_KEY)
    last = ""
    for attempt in range(1, max_retries + 1):
        resp = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        try:
            content = resp.choices[0].message.content or ""
        except Exception:
            content = ""
        if content.strip():
            return content.strip()
        last = content.strip()
        if attempt < max_retries:
            await asyncio.sleep(attempt)
    return last


def _get_or_create_doc_id(client: PageIndexClient, pdf_path: Path) -> str:
    filename = pdf_path.name
    cache = _load_cache()

    if filename in cache:
        cached_id = cache[filename]
        try:
            meta = client.get_document(cached_id)
            if meta.get("status") == "completed":
                return cached_id
        except Exception:
            pass
        del cache[filename]
        _save_cache(cache)

    offset = 0
    while True:
        resp = client.list_documents(limit=100, offset=offset)
        docs = resp.get("documents", [])
        for doc in docs:
            if doc.get("name") == filename and doc.get("status") == "completed":
                doc_id = doc["id"]
                cache[filename] = doc_id
                _save_cache(cache)
                return doc_id
        total = resp.get("total", 0)
        offset += len(docs)
        if offset >= total or not docs:
            break

    result = client.submit_document(str(pdf_path))
    doc_id = result["doc_id"]
    cache[filename] = doc_id
    _save_cache(cache)
    return doc_id


# ──────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    doc_id: str
    question: str


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "pageindex_key": bool(PAGEINDEX_API_KEY), "groq_key": bool(GROQ_API_KEY)}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Save PDF locally, submit to PageIndex, return doc_id and initial status."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    save_path = DOCS_DIR / file.filename
    content = await file.read()
    save_path.write_bytes(content)

    client = _pi_client()
    doc_id = _get_or_create_doc_id(client, save_path)
    meta = client.get_document(doc_id)

    return {"doc_id": doc_id, "filename": file.filename, "status": meta.get("status", "unknown")}


@app.get("/status/{doc_id}")
def get_status(doc_id: str):
    """Return current processing status of a document."""
    client = _pi_client()
    try:
        meta = client.get_document(doc_id)
        ready = client.is_retrieval_ready(doc_id)
        return {"doc_id": doc_id, "status": meta.get("status"), "ready": ready, "meta": meta}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/tree/{doc_id}")
def get_tree(doc_id: str):
    """Fetch the full PageIndex document tree."""
    # Try local cache first
    tree_path = OUTPUT_DIR / doc_id / "tree.json"
    if tree_path.exists():
        return json.loads(tree_path.read_text(encoding="utf-8"))

    client = _pi_client()
    try:
        tree = client.get_tree(doc_id, node_summary=True)["result"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    tree_path.parent.mkdir(parents=True, exist_ok=True)
    tree_path.write_text(json.dumps(tree, indent=2), encoding="utf-8")
    return tree


@app.post("/query")
async def query_document(req: QueryRequest):
    """Run a full RAG cycle: tree search → context extraction → LLM answer."""
    client = _pi_client()
    doc_out = OUTPUT_DIR / req.doc_id

    # ── fetch tree ──────────────────────────────────────────────────
    tree_path = doc_out / "tree.json"
    if tree_path.exists():
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
    else:
        try:
            tree = client.get_tree(req.doc_id, node_summary=True)["result"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Tree fetch failed: {e}")
        doc_out.mkdir(parents=True, exist_ok=True)
        tree_path.write_text(json.dumps(tree, indent=2), encoding="utf-8")

    node_map = utils.create_node_mapping(tree)
    tree_without_text = utils.remove_fields(tree.copy(), fields=["text"])

    # ── node-selection prompt ───────────────────────────────────────
    search_prompt = f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a summary.

Your task is to identify which nodes likely contain the answer.

Question: {req.question}

Document tree:
{json.dumps(tree_without_text, indent=2)}

Reply ONLY in valid JSON:
{{
    "thinking": "",
    "node_list": ["node_id_1", "node_id_2"]
}}
"""
    raw = await _call_llm(search_prompt)
    try:
        search_json = json.loads(_extract_json(raw))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Invalid JSON from model:\n{raw}")

    selected_nodes = search_json.get("node_list", [])
    thinking = search_json.get("thinking", "")

    # ── build context ───────────────────────────────────────────────
    node_details = []
    context_parts = []
    for nid in selected_nodes:
        if nid not in node_map:
            continue
        node = node_map[nid]
        context_parts.append(node.get("text", ""))
        node_details.append({
            "id": nid,
            "title": node.get("title", nid),
            "summary": node.get("summary", ""),
            "text_preview": (node.get("text", "") or "")[:300],
        })

    relevant_content = "\n\n".join(context_parts)

    # ── answer prompt ───────────────────────────────────────────────
    answer_prompt = f"""
Answer the question using ONLY the context below.

Question: {req.question}

Context:
{relevant_content}

Provide a concise answer.
"""
    final_answer = await _call_llm(answer_prompt)

    # ── persist ─────────────────────────────────────────────────────
    doc_out.mkdir(parents=True, exist_ok=True)
    (doc_out / "node_search.json").write_text(
        json.dumps({"query": req.question, **search_json}, indent=2), encoding="utf-8"
    )
    (doc_out / "answer.json").write_text(
        json.dumps({"query": req.question, "answer": final_answer}, indent=2), encoding="utf-8"
    )

    return {
        "answer": final_answer,
        "thinking": thinking,
        "selected_nodes": node_details,
        "doc_id": req.doc_id,
    }


@app.get("/documents")
def list_documents():
    """List all documents (cache + remote API)."""
    cache = _load_cache()
    client = _pi_client()

    remote: list[dict] = []
    try:
        offset = 0
        while True:
            resp = client.list_documents(limit=100, offset=offset)
            docs = resp.get("documents", [])
            remote.extend(docs)
            total = resp.get("total", 0)
            offset += len(docs)
            if offset >= total or not docs:
                break
    except Exception:
        pass

    # Merge cache filename mapping onto remote docs
    id_to_filename = {v: k for k, v in cache.items()}
    for doc in remote:
        if doc.get("id") in id_to_filename:
            doc["cached_filename"] = id_to_filename[doc["id"]]

    return {"documents": remote, "cache": cache}


@app.get("/pdf/{filename}")
def serve_pdf(filename: str):
    """Serve the stored PDF for in-browser preview."""
    pdf_path = DOCS_DIR / filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(str(pdf_path), media_type="application/pdf")
