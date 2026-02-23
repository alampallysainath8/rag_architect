"""
Enhanced RAG — Entry point for CLI and server operations.

Usage:
    # Start the API server
    python main.py serve
    
    # Ingest a document
    python main.py ingest <file_path> <username>
    
    # Search documents
    python main.py search "<query>" <username>
    
    # Ask a question (full RAG)
    python main.py ask "<question>" <username>
"""

import sys
import os
import logging
import uvicorn
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def serve():
    """Start the FastAPI server."""
    print("\n" + "=" * 60)
    print("🚀 Starting Enhanced RAG API Server")
    print("=" * 60)
    
    from app.core.config import settings
    
    print(f"Database: {settings.SQLITE_DB_PATH}")
    print(f"Index: {settings.PINECONE_INDEX_NAME}")
    print(f"Embedding: {settings.EMBEDDING_MODEL}")
    print(f"Reranker: {settings.RERANKER_MODEL}")
    print(f"Server: http://0.0.0.0:8000")
    print(f"Docs: http://0.0.0.0:8000/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


def ingest(file_path: str, username: str):
    """Ingest a document."""
    from app.pipeline.processing_pipeline import ProcessingPipeline
    
    abs_path = os.path.abspath(file_path)
    
    if not os.path.exists(abs_path):
        print(f"❌ File not found: {abs_path}")
        sys.exit(1)
    
    print(f"\n📄 Ingesting: {Path(file_path).name}")
    print(f"👤 User: {username}")
    print("=" * 60)
    
    try:
        pipeline = ProcessingPipeline()
        result = pipeline.process_document(abs_path, username)
        
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        
        for key, value in result.items():
            print(f"{key:20s}: {value}")
        
        if result["status"] == "success":
            print("\n✅ Ingestion completed successfully!")
        elif result["status"] == "duplicate":
            print("\n⚠️  Document is a duplicate")
        else:
            print("\n❌ Ingestion failed")
            sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def search(query: str, username: str, top_k: int = 10):
    """Search documents."""
    from app.pipeline.processing_pipeline import ProcessingPipeline
    
    print(f"\n🔍 Query: {query}")
    print(f"👤 User: {username}")
    print(f"📊 Top K: {top_k}")
    print("=" * 60)
    
    try:
        pipeline = ProcessingPipeline()
        results = pipeline.search_documents(query, username, top_k)
        
        print(f"\n✅ Found {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            print(f"[{i}] Score: {result['score']:.4f}")
            print(f"    Source: {result['source']} (v{result['version']}, p.{result['page_number']})")
            print(f"    Type: {result['content_type']}")
            print(f"    Text: {result['chunk_text'][:150]}...")
            print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def ask(question: str, username: str, use_reranker: bool = True, top_k: int = 10, top_n: int = 5):
    """Ask a question using full RAG pipeline."""
    from app.pipeline.processing_pipeline import ProcessingPipeline
    from app.vectorstore.reranker import Reranker
    from app.generation import Generator
    from app.core.config import settings
    
    print(f"\n💬 Question: {question}")
    print(f"👤 User: {username}")
    print(f"🔀 Reranker: {'enabled' if use_reranker else 'disabled'}")
    print("=" * 60)
    
    try:
        pipeline = ProcessingPipeline()
        
        # Retrieve or rerank
        if use_reranker:
            reranker = Reranker(
                api_key=settings.PINECONE_API_KEY,
                index_name=settings.PINECONE_INDEX_NAME,
                rerank_model=settings.RERANKER_MODEL
            )
            chunks = reranker.rerank(question, username.lower(), top_k, top_n)
            print(f"\n✅ Retrieved and reranked {len(chunks)} chunks\n")
        else:
            chunks = pipeline.search_documents(question, username, top_k)
            print(f"\n✅ Retrieved {len(chunks)} chunks\n")
        
        if not chunks:
            print("❌ No relevant information found")
            return
        
        # Generate answer
        generator = Generator()
        result = generator.generate_with_citations(question, chunks)
        
        print("=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(result["answer"])
        print()
        
        print("=" * 60)
        print("CITATIONS")
        print("=" * 60)
        for citation in result["citations"]:
            print(f"[{citation['number']}] {citation['source']} (v{citation['version']}, p.{citation['page']})")
            print(f"    Type: {citation['content_type']}, Score: {citation['score']:.4f}")
            print(f"    Preview: {citation['text_preview']}")
            print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def show_help():
    """Show help message."""
    print("""
Enhanced RAG — Document ingestion and retrieval with deduplication and versioning

USAGE:
    python main.py <command> [arguments]

COMMANDS:
    serve                           Start the API server
    ingest <file> <username>        Ingest a document
    search "<query>" <username>     Search documents
    ask "<question>" <username>     Ask a question (full RAG)

EXAMPLES:
    # Start server
    python main.py serve
    
    # Ingest a document
    python main.py ingest docs/report.pdf alice
    
    # Search
    python main.py search "revenue growth" alice
    
    # Ask a question
    python main.py ask "What was the revenue in Q4?" alice

For API documentation, start the server and visit:
    http://localhost:8000/docs
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "serve":
        serve()
    
    elif command == "ingest":
        if len(sys.argv) < 4:
            print("❌ Usage: python main.py ingest <file_path> <username>")
            sys.exit(1)
        ingest(sys.argv[2], sys.argv[3])
    
    elif command == "search":
        if len(sys.argv) < 4:
            print("❌ Usage: python main.py search \"<query>\" <username>")
            sys.exit(1)
        search(sys.argv[2], sys.argv[3])
    
    elif command == "ask":
        if len(sys.argv) < 4:
            print("❌ Usage: python main.py ask \"<question>\" <username>")
            sys.exit(1)
        ask(sys.argv[2], sys.argv[3])
    
    else:
        print(f"❌ Unknown command: {command}")
        show_help()
        sys.exit(1)
