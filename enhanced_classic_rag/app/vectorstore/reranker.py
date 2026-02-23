"""
Reranker — uses Pinecone's hosted reranker to improve result relevance.

Features:
- Integrated search + rerank in single call
- Configurable reranking model
- Top-N selection after reranking
"""

import logging
from typing import List, Dict, Any
from pinecone import Pinecone

logger = logging.getLogger(__name__)


class Reranker:
    """Handles result reranking using Pinecone's integrated reranker."""
    
    def __init__(
        self,
        api_key: str,
        index_name: str,
        rerank_model: str = "bge-reranker-v2-m3"
    ):
        """
        Initialize reranker.
        
        Args:
            api_key: Pinecone API key.
            index_name: Name of the Pinecone index.
            rerank_model: Reranking model name.
        """
        self.api_key = api_key
        self.index_name = index_name
        self.rerank_model = rerank_model
        
        self._pc = Pinecone(api_key=api_key)
        self._index = None
    
    def _get_index(self):
        """Get the Pinecone index."""
        if self._index is None:
            self._index = self._pc.Index(self.index_name)
        return self._index
    
    def rerank(
        self,
        query: str,
        namespace: str,
        top_k: int = 10,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search and rerank results in a single call.
        
        Args:
            query: Search query.
            namespace: Namespace to search in.
            top_k: Number of initial candidates to retrieve.
            top_n: Number of top results to return after reranking.
        
        Returns:
            List of reranked results with scores and metadata.
        """
        index = self._get_index()
        
        try:
            results = index.search(
                namespace=namespace,
                query={
                    "top_k": top_k,
                    "inputs": {"text": query},
                },
                rerank={
                    "model": self.rerank_model,
                    "top_n": top_n,
                    "rank_fields": ["chunk_text"],
                },
                fields=["chunk_text", "source", "page_number", "content_type", "version", "date"],
            )
            
            hits = []
            for item in results.get("result", {}).get("hits", []):
                hits.append({
                    "id": item.get("_id", ""),
                    "score": item.get("_score", 0.0),
                    "chunk_text": item.get("fields", {}).get("chunk_text", ""),
                    "source": item.get("fields", {}).get("source", ""),
                    "page_number": item.get("fields", {}).get("page_number", 0),
                    "content_type": item.get("fields", {}).get("content_type", ""),
                    "version": item.get("fields", {}).get("version", 1),
                    "date": item.get("fields", {}).get("date", ""),
                })
            
            logger.info(
                f"Reranked {len(hits)} results (from {top_k} candidates) "
                f"for query in namespace '{namespace}'"
            )
            return hits
            
        except Exception as e:
            logger.error(f"Error reranking: {e}")
            raise


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("=== Reranker Test ===\n")
    
    # Test configuration
    api_key = os.getenv("PINECONE_API_KEY", "")
    
    if not api_key:
        print("❌ PINECONE_API_KEY not set. Skipping test.")
    else:
        try:
            reranker = Reranker(
                api_key=api_key,
                index_name="enhanced-rag-test",
                rerank_model="bge-reranker-v2-m3"
            )
            
            # Test reranking
            results = reranker.rerank(
                query="artificial intelligence",
                namespace="testuser",
                top_k=10,
                top_n=3
            )
            
            print(f"✅ Reranked results: {len(results)}")
            
            for i, result in enumerate(results, 1):
                print(f"\n[{i}] Score: {result['score']:.4f}")
                print(f"    Source: {result['source']}")
                print(f"    Text: {result['chunk_text'][:100]}...")
            
            print("\n✅ Test passed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
