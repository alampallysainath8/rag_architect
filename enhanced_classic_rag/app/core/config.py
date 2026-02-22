"""
Configuration module — centralized settings for the Enhanced RAG system.
All configuration values loaded from environment variables or defaults.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Centralized configuration settings."""
    
    # ── Database Settings ────────────────────────────────────────────────
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "rag_metadata.db")
    
    # ── API Keys ─────────────────────────────────────────────────────────
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # ── Pinecone Settings ────────────────────────────────────────────────
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "rag-classic")
    PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
    DEFAULT_NAMESPACE: str = os.getenv("DEFAULT_NAMESPACE", "default")
    
    # ── Embedding Settings ───────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "multilingual-e5-large")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    
    # ── Chunking Settings ────────────────────────────────────────────────
    TEXT_CHUNK_SIZE: int = int(os.getenv("TEXT_CHUNK_SIZE", "500"))
    TEXT_CHUNK_OVERLAP: int = int(os.getenv("TEXT_CHUNK_OVERLAP", "80"))
    TABLE_CHUNK_SIZE: int = int(os.getenv("TABLE_CHUNK_SIZE", "800"))
    
    # ── Retrieval Settings ───────────────────────────────────────────────
    TOP_K: int = int(os.getenv("TOP_K", "10"))
    
    # ── Reranker Settings ────────────────────────────────────────────────
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "bge-reranker-v2-m3")
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "5"))
    
    # ── Generation Settings ──────────────────────────────────────────────
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
    
    # ── Batch Settings ───────────────────────────────────────────────────
    UPSERT_BATCH_SIZE: int = int(os.getenv("UPSERT_BATCH_SIZE", "96"))
    
    def validate(self) -> None:
        """Validate required configuration values."""
        if not self.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is required")
        if not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required")


# Global settings instance
settings = Settings()


if __name__ == "__main__":
    print("=== Enhanced RAG Configuration ===")
    print(f"Database path     : {settings.SQLITE_DB_PATH}")
    print(f"PINECONE_API_KEY  : {'✅ set' if settings.PINECONE_API_KEY else '❌ missing'}")
    print(f"GROQ_API_KEY      : {'✅ set' if settings.GROQ_API_KEY else '❌ missing'}")
    print(f"Index name        : {settings.PINECONE_INDEX_NAME}")
    print(f"Embed model       : {settings.EMBEDDING_MODEL}")
    print(f"Rerank model      : {settings.RERANKER_MODEL}")
    print(f"LLM model         : {settings.GROQ_MODEL}")
    print(f"Text chunk size   : {settings.TEXT_CHUNK_SIZE} (overlap: {settings.TEXT_CHUNK_OVERLAP})")
    print(f"Table chunk size  : {settings.TABLE_CHUNK_SIZE}")
    print(f"Retrieval TOP_K   : {settings.TOP_K}")
    print(f"Rerank TOP_N      : {settings.RERANK_TOP_N}")
    
    try:
        settings.validate()
        print("\n✅ Configuration validated successfully!")
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
