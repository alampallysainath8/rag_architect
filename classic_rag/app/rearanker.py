"""
Reranker module — uses Pinecone's hosted BGE reranker to reorder
retrieval results by relevance to the query.
"""

from typing import List, Dict
from pinecone import Pinecone

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_RERANK_MODEL,
    RERANK_TOP_N,
    TOP_K,
)

_pc = Pinecone(api_key=PINECONE_API_KEY)


def rerank(query: str, top_k: int = TOP_K, top_n: int = RERANK_TOP_N) -> List[Dict]:
    """
    Search + rerank in a single Pinecone call.
    Uses the integrated reranking API with `bge-reranker-v2-m3`.

    Returns a list of the top_n most relevant chunks.
    """
    index = _pc.Index(PINECONE_INDEX_NAME)

    reranked = index.search(
        namespace=PINECONE_NAMESPACE,
        query={
            "top_k": top_k,
            "inputs": {"text": query},
        },
        rerank={
            "model": PINECONE_RERANK_MODEL,
            "top_n": top_n,
            "rank_fields": ["chunk_text"],
        },
        fields=["chunk_text", "source", "pages"],
    )

    hits = []
    for item in reranked.get("result", {}).get("hits", []):
        hits.append(
            {
                "id": item.get("_id", ""),
                "score": item.get("_score", 0.0),
                "chunk_text": item.get("fields", {}).get("chunk_text", ""),
                "source": item.get("fields", {}).get("source", ""),
                "pages": item.get("fields", {}).get("pages", ""),
            }
        )

    return hits


if __name__ == "__main__":
    import sys

    print("=== Reranker Test ===")
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What was Apple's revenue?"
    print(f"Query: {query}\n")

    results = rerank(query, top_k=5, top_n=3)
    print(f"Reranked {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] score={r['score']:.4f} | source={r['source']}")
        print(f"      {r['chunk_text'][:150]}...\n")
    print("✅ Reranker test passed!")