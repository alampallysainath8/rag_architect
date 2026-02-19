"""
Configuration module — loads environment variables and defines app-wide settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ── Pinecone Settings ────────────────────────────────────────────────────────
PINECONE_INDEX_NAME: str = "rag-classic"
PINECONE_NAMESPACE: str = "documents"
PINECONE_CLOUD: str = "aws"
PINECONE_REGION: str = "us-east-1"
PINECONE_EMBED_MODEL: str = "multilingual-e5-large"  # Pinecone hosted embedding
PINECONE_RERANK_MODEL: str = "bge-reranker-v2-m3"    # Pinecone hosted reranker

# ── Chunking Settings ────────────────────────────────────────────────────────
CHUNK_SIZE: int = 512          # characters per chunk
CHUNK_OVERLAP: int = 64        # overlap between chunks

# ── Retrieval Settings ────────────────────────────────────────────────────────
TOP_K: int = 10                # candidates to fetch from vector search
RERANK_TOP_N: int = 5          # results to keep after reranking

# ── Generation Settings ──────────────────────────────────────────────────────
OPENAI_MODEL: str = "gpt-4o-mini"
GROQ_MODEL: str = "openai/gpt-oss-20b"
MAX_TOKENS: int = 1024
TEMPERATURE: float = 0.2


if __name__ == "__main__":
    print("=== Config Test ===")
    print(f"PINECONE_API_KEY : {'✅ set' if PINECONE_API_KEY else '❌ missing'}")
    print(f"OPENAI_API_KEY   : {'✅ set' if OPENAI_API_KEY else '❌ missing'}")
    print(f"GROQ_API_KEY     : {'✅ set' if GROQ_API_KEY else '❌ missing'}")
    print(f"Index name       : {PINECONE_INDEX_NAME}")
    print(f"Embed model      : {PINECONE_EMBED_MODEL}")
    print(f"Rerank model     : {PINECONE_RERANK_MODEL}")
    print(f"Chunk size       : {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
    print(f"LLM model        : {OPENAI_MODEL}")
    print("✅ Config loaded successfully!")