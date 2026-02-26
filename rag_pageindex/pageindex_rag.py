"""
Simple Vectorless RAG with PageIndex using Groq
(Local PDF version)
"""

import os
import json
import asyncio
import time
from pathlib import Path
import dotenv

# doc_id cache lives next to this script
_CACHE_FILE = Path(__file__).parent / "doc_id_cache.json"


def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def get_or_create_doc_id(client: "PageIndexClient", pdf_path: Path) -> str:
    """
    Return the existing doc_id for the given PDF if one is already indexed
    and ready, otherwise submit it fresh.

    Resolution order:
      1. Local cache (doc_id_cache.json) – zero API calls
      2. PageIndex list_documents API – scans remote docs by filename
      3. Submit new document
    """
    filename = pdf_path.name
    cache = _load_cache()

    # ── 1. Local cache hit ──────────────────────────────────────────────
    if filename in cache:
        cached_id = cache[filename]
        try:
            meta = client.get_document(cached_id)
            if meta.get("status") == "completed":
                print(f"Reusing cached doc_id for '{filename}': {cached_id}")
                return cached_id
            else:
                print(
                    f"Cached doc_id '{cached_id}' status={meta.get('status')}, ignoring."
                )
        except Exception as e:
            print(f"Cached doc_id '{cached_id}' no longer valid ({e}), ignoring.")
        del cache[filename]
        _save_cache(cache)

    # ── 2. Search remote document list by filename ──────────────────────
    print(f"Checking PageIndex API for existing index of '{filename}'...")
    offset = 0
    while True:
        resp = client.list_documents(limit=100, offset=offset)
        docs = resp.get("documents", [])
        for doc in docs:
            if doc.get("name") == filename and doc.get("status") == "completed":
                doc_id = doc["id"]
                print(f"Found existing remote index for '{filename}': {doc_id}")
                cache[filename] = doc_id
                _save_cache(cache)
                return doc_id
        total = resp.get("total", 0)
        offset += len(docs)
        if offset >= total or not docs:
            break

    # ── 3. Submit fresh ─────────────────────────────────────────────────
    print(f"No existing index found for '{filename}'. Submitting new document...")
    result = client.submit_document(str(pdf_path))
    doc_id = result["doc_id"]
    cache[filename] = doc_id
    _save_cache(cache)
    return doc_id

dotenv.load_dotenv()

from pageindex import PageIndexClient
import pageindex.utils as utils
from groq import AsyncGroq

# -------------------------
# CONFIGURATION
# -------------------------

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "llama-3.3-70b-versatile"

# 👇 Provide your local PDF path here
PDF_PATH = r"C:\Users\Desktop\lang_proj\rag_architect\rag_pageindex\1706.03762v7-1-5.pdf"

# -------------------------
# VALIDATE FILE
# -------------------------

pdf_file = Path(PDF_PATH)

if not pdf_file.exists():
    raise FileNotFoundError(f"PDF not found at: {PDF_PATH}")

print(f"Using local PDF: {pdf_file.resolve()}")

# -------------------------
# PAGEINDEX CLIENT
# -------------------------

pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

# -------------------------
# GROQ LLM CALL
# -------------------------

async def call_llm(prompt, temperature=0, max_retries=3):
    client = AsyncGroq(api_key=GROQ_API_KEY)

    last_content = ""
    for attempt in range(1, max_retries + 1):
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        content = None
        try:
            content = response.choices[0].message.content
        except Exception:
            content = None

        if content:
            last_content = content.strip()
            if last_content:
                return last_content

        # small backoff before retrying
        if attempt < max_retries:
            await asyncio.sleep(attempt)

    # Return whatever we have (may be empty string)
    return last_content


# -------------------------
# GET OR CREATE DOCUMENT INDEX
# -------------------------

doc_id = get_or_create_doc_id(pi_client, pdf_file)
print("Using doc_id:", doc_id)

# -------------------------
# OUTPUT FOLDER
# -------------------------

OUTPUT_DIR = Path(__file__).parent / "output" / doc_id
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output folder: {OUTPUT_DIR}")

# -------------------------
# WAIT FOR INDEX GENERATION
# -------------------------

print("Waiting for PageIndex tree to become ready...")
while not pi_client.is_retrieval_ready(doc_id):
    print("Processing... sleeping for 5 seconds")
    time.sleep(5)

print("Index ready ✅")

# -------------------------
# FETCH TREE
# -------------------------

tree = pi_client.get_tree(doc_id, node_summary=True)["result"]
node_map = utils.create_node_mapping(tree)

print("Tree fetched successfully!")

tree_path = OUTPUT_DIR / "tree.json"
tree_path.write_text(json.dumps(tree, indent=2), encoding="utf-8")
print(f"Tree saved → {tree_path}")

# -------------------------
# QUERY
# -------------------------

query = "What are the conclusions in this document?"

tree_without_text = utils.remove_fields(tree.copy(), fields=["text"])

search_prompt = f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a summary.

Your task is to identify which nodes likely contain the answer.

Question: {query}

Document tree:
{json.dumps(tree_without_text, indent=2)}

Reply ONLY in valid JSON:
{{
    "thinking": "",
    "node_list": ["node_id_1", "node_id_2"]
}}
"""

# -------------------------
# TREE SEARCH (REASONING)
# -------------------------

tree_search_result = asyncio.run(call_llm(search_prompt))


def extract_json(text: str) -> str:
    """Strip markdown code fences and return the raw JSON string."""
    import re
    # Remove ```json ... ``` or ``` ... ``` wrappers
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


try:
    tree_search_json = json.loads(extract_json(tree_search_result))
except json.JSONDecodeError:
    raise ValueError(f"Invalid JSON from model:\n{tree_search_result}")

node_search_path = OUTPUT_DIR / "node_search.json"
node_search_path.write_text(
    json.dumps({"query": query, **tree_search_json}, indent=2), encoding="utf-8"
)
print(f"Node search saved → {node_search_path}")

print("\nRetrieved Nodes:")
for node_id in tree_search_json["node_list"]:
    print(f"  • {node_id} – {node_map[node_id]['title']}")

# -------------------------
# CONTEXT EXTRACTION
# -------------------------

relevant_content = "\n\n".join(
    node_map[node_id]["text"] for node_id in tree_search_json["node_list"]
)

answer_prompt = f"""
Answer the question using ONLY the context below.

Question: {query}

Context:
{relevant_content}

Provide a concise answer.
"""

final_answer = asyncio.run(call_llm(answer_prompt))

answer_path = OUTPUT_DIR / "answer.json"
answer_path.write_text(
    json.dumps({"query": query, "answer": final_answer}, indent=2), encoding="utf-8"
)
print(f"Answer saved → {answer_path}")

print("\n===== FINAL ANSWER =====")
print(final_answer)
print("========================")